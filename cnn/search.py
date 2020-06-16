#!/usr/bin/python
# -*- coding: utf-8 -*-

# basic
import argparse
import os
import socket
import random
import shutil
import time
import warnings
import sys
import copy
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
from tensorboardX import SummaryWriter

# darts
from config import SearchConfig
from models.search_cnn import SearchCNNController
import tools.utils as utils
from tools.architect import Architect
from tools.visualize import plot

# dp
from tools.gradient_clipping import clipping_dispatcher

def find_free_port():
    # import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def main():
    # init config
    config = SearchConfig()

    # set seed
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! ')

    # For slurm available
    if config.world_size == -1 and "SLURM_NPROCS" in os.environ:
        # acquire world size from slurm
        config.world_size = int(os.environ["SLURM_NPROCS"])
        config.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        hostfile = os.path.join(config.dist_path, "dist_url." + jobid  + ".txt")
        if config.dist_file is not None:
            config.dist_url = "file://{}.{}".format(os.path.realpath(config.dist_file), jobid)
        elif config.rank == 0:
            if config.dist_backend == 'nccl' and config.infi_band:
                # only NCCL backend supports inifiniband
                interface_str = 'ib{:d}'.format(config.infi_band_interface)
                print("Use infiniband support on interface " + interface_str + '.')
                os.environ['NCCL_SOCKET_IFNAME'] = interface_str
                os.environ['GLOO_SOCKET_IFNAME'] = interface_str
                ip_str = os.popen('ip addr show ' + interface_str).read()
                ip = ip_str.split("inet ")[1].split("/")[0]
            else:
                if config.world_size == 1:  # use only one node
                    ip = '127.0.0.1'
                else:
                    ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            config.dist_url = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w") as f:
                f.write(config.dist_url)
        else:
            while not os.path.exists(hostfile):
                time.sleep(5)  # waite for the main process
            with open(hostfile, "r") as f:
                config.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(config.dist_url, config.rank, config.world_size))

    # support multiple GPU on one node
    # assume each node have equal GPUs
    ngpus_per_node = torch.cuda.device_count()
    if config.mp_dist:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # worker process function
        mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call worker function on first GPU device
        worker(None, ngpus_per_node, config)


def worker(gpu, ngpus_per_node, config_in):
    # init
    config = copy.deepcopy(config_in)
    jobid = os.environ["SLURM_JOBID"]
    procid = int(os.environ["SLURM_PROCID"])
    config.gpu = gpu

    if config.gpu is not None:
        writer_name = "tb.{}-{:d}-{:d}".format(jobid, procid, gpu)
        logger_name = "{}.{}-{:d}-{:d}.search.log".format(config.name, jobid, procid, gpu)
        ploter_name = "{}-{:d}-{:d}".format(jobid, procid, gpu)
        ck_name = "{}-{:d}-{:d}".format(jobid, procid, gpu)
    else:
        writer_name = "tb.{}-{:d}-all".format(jobid, procid)
        logger_name = "{}.{}-{:d}-all.search.log".format(config.name, jobid, procid)
        ploter_name = "{}-{:d}-all".format(jobid, procid)
        ck_name = "{}-{:d}-all".format(jobid, procid)

    writer = SummaryWriter(log_dir=os.path.join(config.path, writer_name))
    writer.add_text('config', config.as_markdown(), 0)
    logger = utils.get_logger(os.path.join(config.path, logger_name))

    config.print_params(logger.info)

    # get cuda device
    device = torch.device('cuda', gpu)

    # begin
    logger.info("Logger is set - training start")

    if config.dist_url == "env://" and config.rank == -1:
        config.rank = int(os.environ["RANK"])

    if config.mp_dist:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        config.rank = config.rank * ngpus_per_node + gpu
    # print('back:{}, dist_url:{}, world_size:{}, rank:{}'.format(config.dist_backend, config.dist_url, config.world_size, config.rank))
    dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                            world_size=config.world_size, rank=config.rank)

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    # build model
    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit)
    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        # model = model.to(device)
        model.cuda(config.gpu)
        # When using a single GPU per process and per DistributedDataParallel, we need to divide
        # the batch size ourselves based on the total number of GPUs we have
        config.batch_size = int(config.batch_size / ngpus_per_node)
        config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.rank])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)

    # weights optimizer
    w_optim = torch.optim.SGD(model.module.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.module.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_data_ = data.Subset(train_data, indices[:split])
    valid_data_ = data.Subset(train_data, indices[split:])
    train_sampler = data.distributed.DistributedSampler(train_data_,
                                                        num_replicas=config.world_size,
                                                        rank=config.rank)
    valid_sampler = data.distributed.DistributedSampler(valid_data_,
                                                        num_replicas=config.world_size,
                                                        rank=config.rank)
    train_loader = data.DataLoader(train_data_,
                                   batch_size=config.batch_size,
                                   sampler=train_sampler,
                                   shuffle=False,
                                   num_workers=config.workers,
                                   pin_memory=True)
    valid_loader = data.DataLoader(valid_data_,
                                   batch_size=config.batch_size,
                                   sampler=valid_sampler,
                                   shuffle=False,
                                   num_workers=config.workers,
                                   pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # setting the privacy protecting procedure
    if config.dist_privacy:
        logger.info("PRIVACY ENGINE OFF")

    # training loop
    best_top1 = 0.0
    for epoch in range(config.epochs):
        # lr_scheduler.step()
        # lr = lr_scheduler.get_lr()[0]
        lr = lr_scheduler.get_last_lr()[0]

        model.module.print_alphas(logger)

        # training
        train(logger, writer, device, config,
              train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)
        lr_scheduler.step()  # move to the place after optimizer.step()

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(logger, writer, device, config,
                        valid_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.module.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "JOB" + ploter_name + "-EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, ck_name, 'search', is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def train(logger, writer, device, config, train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(config, trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.module.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.module.weights(), config.w_grad_clip)
        if config.dist_privacy:
            # privice gradient clipping
            clipping_dispatcher(model.module.named_weights(),
                                config.max_weights_grad_norm,
                                config.var_gamma,
                                device,
                                logger
                                )
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(logger, writer, device, config, valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.module.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()