""" Config class for search/augment """
import argparse
import os
from tools import genotypes as gt
from functools import partial
import torch


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        # parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
        #                     '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')
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

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self._mk_folder('searchs')
        self.path = os.path.join('searchs', self.name)
        self._mk_folder(self.path)
        self.plot_path = os.path.join(self.path, 'plots')
        self._mk_folder(self.plot_path)
        self.dist_path = os.path.join(self.path, 'dist')
        self._mk_folder(self.dist_path)
        # self.gpus = parse_gpus(self.gpus)

    def _mk_folder(self, path_in):
        if not os.path.exists(path_in):
            os.mkdir(os.path.abspath(path_in))


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
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

        parser.add_argument('--genotype', required=True, help='Cell genotype')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self._mk_folder('augments')
        self.path = os.path.join('augments', self.name)
        self._mk_folder(self.path)
        self.dist_path = os.path.join(self.path, 'dist')
        self._mk_folder(self.dist_path)

        self.genotype = gt.from_str(self.genotype)
        self.gpus = parse_gpus(self.gpus)

    def _mk_folder(self, path_in):
        if not os.path.exists(path_in):
            os.mkdir(os.path.abspath(path_in))