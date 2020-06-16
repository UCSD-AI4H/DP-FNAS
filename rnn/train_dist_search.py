import argparse
import os, sys, glob, copy
import socket
import time
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as torchdata

from architect import Architect

import gc

import data
import model_search

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint, get_logger
from gradient_clipping import clipping_dispatcher

from tensorboardX import SummaryWriter


def evaluate(model, corpus, args_in, data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    args = args_in
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.module.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            targets = targets.view(-1)
            log_prob, hidden = model.module(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
            total_loss += loss * len(data)
            hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(model, architect, epoch, corpus, train_data, search_data, optimizer, device, logger, writer, args_in):
    args = args_in
    # print("batch_size:", args.batch_size, "small_batch_size:", args.small_batch_size)
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.module.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    hidden_valid = [model.module.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        data_valid, targets_valid = get_batch(search_data, i % (search_data.size(0) - 1), args)
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
            cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])
            hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

            hidden_valid[s_id], grad_norm = architect.step(
                    hidden[s_id], cur_data, cur_targets,
                    hidden_valid[s_id], cur_data_valid, cur_targets_valid,
                    optimizer,
                    args.unrolled)

            # assuming small_batch_size = batch_size so we don't accumulate gradients
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = model.module(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
              loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip)
        if args.dist_privacy:
            # privice gradient clipping
            clipping_dispatcher(model.module.named_parameters(),
                                args.max_weights_grad_norm,
                                args.var_gamma,
                                device,
                                logger
                                )
        optimizer.step()

        if batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            writer.add_scalar('train/loss', cur_loss, batch)
            writer.add_scalar('train/ppl', math.exp(cur_loss), batch)

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            logger.info(model.module.genotype())
            # print(F.softmax(parallel_model.weights, dim=-1))
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        batch += 1
        i += seq_len



def find_free_port():
    # import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def main(args_in):
    config = args_in

    # set path
    def _mk_folder(path_in):
        if not os.path.exists(path_in):
            Path(path_in).mkdir(parents=True, exist_ok=True)
            # os.mkdir(os.path.abspath(path_in))

    # if not config.continue_train:
    #     root_folder = 'searchs'
    #     _mk_folder('./' + root_folder)
    #     sub_folder = 'eval-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    #     config.save = os.path.join(root_folder, sub_folder)
    #     create_exp_dir(config.save, scripts_to_save=glob.glob('*.py'))
    if not config.continue_train:
        root_folder = 'searchs'
        _mk_folder('./' + root_folder)
        if "SLURM_NPROCS" in os.environ:
            sub_folder = 'eval-{}-{}-{}'.format(config.save, time.strftime("%Y%m%d"), os.environ["SLURM_JOBID"])
        else:
            sub_folder = 'eval-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
        config.save = os.path.join(root_folder, sub_folder)
        _mk_folder(config.save)
        # if not os.path.exists(config.save):
        #     create_exp_dir(config.save, scripts_to_save=glob.glob('*.py'))
    config.path = config.save
    config.dist_path = os.path.join(config.path, 'dist')
    _mk_folder(config.dist_path)


    # set seed
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        cudnn.deterministic = True

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
                if config.world_size == 1:  # use only one node (localhost)
                    ip = '127.0.0.1'
                else:
                    ip = socket.gethostbyname(socket.gethostname())
                os.environ['NCCL_SOCKET_IFNAME'] = 'bond0'
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
    args = config
    jobid = os.environ["SLURM_JOBID"]
    procid = int(os.environ["SLURM_PROCID"])
    config.gpu = gpu

    if config.gpu is not None:
        writer_name = "tb.{}-{:d}-{:d}".format(jobid, procid, gpu)
        logger_name = "{}.{}-{:d}-{:d}.search.log".format(config.name, jobid, procid, gpu)
        model_name = "{}-{:d}-{:d}-model.pt".format(jobid, procid, gpu)
        optimizer_name = "{}-{:d}-{:d}-optimizer.pt".format(jobid, procid, gpu)
        msic_name = "{}-{:d}-{:d}-misc.pt".format(jobid, procid, gpu)
        ck_name = "{}-{:d}-{:d}".format(jobid, procid, gpu)
    else:
        writer_name = "tb.{}-{:d}-all".format(jobid, procid)
        logger_name = "{}.{}-{:d}-all.search.log".format(config.name, jobid, procid)
        model_name = "{}-{:d}-all-model.pt".format(jobid, procid)
        optimizer_name = "{}-{:d}-all-optimizer.pt".format(jobid, procid)
        msic_name = "{}-{:d}-all-misc.pt".format(jobid, procid)
        ck_name = "{}-{:d}-all".format(jobid, procid)

    writer = SummaryWriter(log_dir=os.path.join(config.path, writer_name))
    # writer.add_text('config', config.as_markdown(), 0)
    logger = get_logger(os.path.join(config.path, logger_name))

    # get cuda device
    device = torch.device('cuda', gpu)

    # ==============================  begin  ==============================
    logger.info("Logger is set - training start")
    logger.info('Args: {}'.format(args))

    if config.dist_url == "env://" and config.rank == -1:
        config.rank = int(os.environ["RANK"])

    if config.mp_dist:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        config.rank = config.rank * ngpus_per_node + gpu
    # print('back:{}, dist_url:{}, world_size:{}, rank:{}'.format(config.dist_backend, config.dist_url, config.world_size, config.rank))
    dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                            world_size=config.world_size, rank=config.rank)

    # get data
    corpus = data.Corpus(args.data)

    eval_batch_size = 10
    test_batch_size = 1

    train_data = batchify(corpus.train, args.batch_size, args)
    search_data = batchify(corpus.valid, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    # split data ( with respect to GPU_id)
    def split_set(set_in):
        per_set_length = set_in.size(0) // config.world_size
        set_out = set_in[per_set_length*config.rank + 0: per_set_length*config.rank + per_set_length]
        return set_out
    train_data = split_set(train_data).to(device)
    search_data = split_set(search_data).to(device)
    val_data = split_set(val_data).to(device)
    test_data = split_set(test_data).to(device)

    if config.dist_privacy:
        logger.info("PRIVACY ENGINE ON")

    # build model
    ntokens = len(corpus.dictionary)
    if args.continue_train:
        model = torch.load(os.path.join(args.save, model_name))
    else:
        model = model_search.RNNModelSearch(ntokens, args.emsize, args.nhid, args.nhidlast,
                        args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute)
    # make model distributed
    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        # model = model.to(device)
        model.cuda(config.gpu)
        # When using a single GPU per process and per DistributedDataParallel, we need to divide
        # the batch size ourselves based on the total number of GPUs we have
        # config.batch_size = int(config.batch_size / ngpus_per_node)
        config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.rank])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)

    architect = Architect(model.module, args)

    total_params = sum(x.data.nelement() for x in model.module.parameters())
    logger.info('Model total parameters: {}'.format(total_params))

    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000

    if args.continue_train:
        optimizer_state = torch.load(os.path.join(args.save, optimizer_name))
        if 't0' in optimizer_state['param_groups'][0]:
            optimizer = torch.optim.ASGD(model.module.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            optimizer = torch.optim.SGD(model.module.parameters(), lr=args.lr, weight_decay=args.wdecay)
        optimizer.load_state_dict(optimizer_state)
    else:
        optimizer = torch.optim.SGD(model.module.parameters(), lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        # train()
        train(model, architect, epoch, corpus, train_data, search_data, optimizer,
              device, logger, writer, args)

        val_loss = evaluate(model, corpus, args, val_data, eval_batch_size)
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        logger.info('-' * 89)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/ppl', math.exp(val_loss), epoch)

        if val_loss < stored_loss:
            save_checkpoint(model, optimizer, epoch, args.save, dist_name=ck_name)
            logger.info('Saving Normal!')
            stored_loss = val_loss

        best_val_loss.append(val_loss)

    test_loss = evaluate(model, corpus, args, test_data, test_batch_size)
    logger.info('=' * 89)
    logger.info('| End of training & Testing | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)



if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model (distributed)')
    # import
    parser.add_argument('--name', required=True)
    parser.add_argument('--workers', type=int, default=4, help='# of workers')
    # original
    parser.add_argument('--data', type=str, default='../data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nhidlast', type=int, default=300,
                        help='number of hidden units for the last rnn layer')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.75,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.25,
                        help='dropout for hidden nodes in rnn layers (0 = no dropout)')
    parser.add_argument('--dropoutx', type=float, default=0.75,
                        help='dropout for input nodes in rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.2,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=3,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='EXP',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=0,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=5e-7,
                        help='weight decay applied to all weights')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue train from a checkpoint')
    parser.add_argument('--small_batch_size', type=int, default=-1,
                        help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                        In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                        until batch_size is reached. An update step is then performed.')
    parser.add_argument('--max_seq_len_delta', type=int, default=20,
                        help='max sequence length')
    # parser.add_argument('--gpu', type=str, default='0', help='GPU device to use')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                        help='weight decay for the architecture encoding alpha')
    parser.add_argument('--arch_lr', type=float, default=3e-3,
                        help='learning rate for the architecture encoding alpha')
    # distributed training
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend (default nccl)')
    parser.add_argument('--infi_band', type=str2bool, default=True, help='use infiniband')
    parser.add_argument('--infi_band_interface', default=0, type=int, help='default infiniband interface id')
    parser.add_argument('--world_size', type=int, default=-1, help='# of computation node')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_file', default=None, type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--mp_dist', type=str2bool, default=True, help='allow multiple GPU on 1 node')
    parser.add_argument('--gpu', default=None, type=int, help='local GPU id to use')
    # privacy protect
    parser.add_argument('--dist_privacy', type=str2bool, default=False, help='use gassian noise to enhance privacy protecting (default off)')
    parser.add_argument('--var_sigma', default=1.0, type=float, help='the varian of gassian noise on A')
    parser.add_argument('--var_gamma', default=1.0, type=float, help='the varian of gassian noise on W')
    parser.add_argument('--max_hessian_grad_norm', default=2.0, type=float, help='Clip alpha gradients to this norm (default 2.0)')
    parser.add_argument('--max_weights_grad_norm', default=2.0, type=float, help='Clip alpha gradients to this norm (default 2.0)')

    args = parser.parse_args()

    if args.nhidlast < 0:
        args.nhidlast = args.emsize
    if args.small_batch_size < 0:
        args.small_batch_size = args.batch_size

    main(args)