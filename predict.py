# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import pickle
import torch
import argparse

from models import *
from pruners import predictive
from dataset.weight_initializers import init_net

from dataset.dataset import get_dataloaders
from utils import get_net_from_config


def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for Architecture')

    parser.add_argument('--outdir', default='./result',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=5, help='frequency of write to file')
    parser.add_argument('--arch_name', default='darts',
                        help='architecture name in [darts, benchmark201]')
    parser.add_argument('--train_history_file_name', default='result/darts_train_cf10_r8_c16_e40_bz128.p',
                        help='train history file name')

    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


def get_num_classes(dataset_name):
    return 100 if dataset_name == 'cifar100' else 10 if dataset_name == 'cifar10' else 120


def load_arch(file_name):
    train_history = {}
    f = open(file_name, 'rb')
    while (1):
        try:
            each_history = pickle.load(f)
            train_history[each_history['idx']] = each_history
        except EOFError:
            break
    f.close()
    return train_history


if __name__ == '__main__':
    args = parse_arguments()
    file_name = args.train_history_file_name

    num_class = get_num_classes(args.dataset)
    train_history = load_arch(file_name)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader = get_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                     args.num_data_workers)

    cached_res = []
    pre = 'cf' if 'cifar' in args.dataset else 'im'
    pfn = f'{args.arch_name}_{pre}{get_num_classes(args.dataset)}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}.p'
    op = os.path.join(args.outdir, pfn)

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    # loop over archs
    len_train_history = len(train_history.keys())
    for i, idx in enumerate(train_history.keys()):

        arch_config = train_history[idx]['config']
        res = {'i': idx, 'arch': arch_config}

        net = get_net_from_config(args.arch_name, arch_config, num_class)
        net.to(args.device)
        init_net(net, args.init_w_type, args.init_b_type)

        measures = predictive.find_measures(net,
                                            train_loader,
                                            (args.dataload, args.dataload_info, get_num_classes(args.dataset)),
                                            args.device)

        res['log_measures'] = measures

        res['val_acc'] = train_history[idx]['log_measures'][-1]['val_acc']

        # print(res)
        cached_res.append(res)

        # write to file
        print(i, len_train_history)
        if i % args.write_freq == 0 or i == (len_train_history - 1):
            print(f'writing {len(cached_res)} results to {op}')
            pf = open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []
