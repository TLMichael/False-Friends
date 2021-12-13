import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import argparse
import os
from tqdm import tqdm
from pprint import pprint

from utils import set_seed, infer_arch, cifar10_class, make_and_restore_cifar_model, PoisonDataset
from make_data import generate_naive, generate_noise, generate_mislabeling, generate_poisoning, visualize


def main(args):
    
    if os.path.isfile(args.poison_file_path):
        print('Poison [{}] already exists.'.format(args.data_type))
        return
    
    data_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())
    data_loader = DataLoader(data_set, batch_size=256, shuffle=False)

    set_seed(args.seed)
    if args.data_type == 'Naive':
        poison_data = generate_naive(args, data_loader)
    elif args.data_type == 'Noise':
        poison_data = generate_noise(args, data_loader)
    elif args.data_type == 'Mislabeling':
        poison_data = generate_mislabeling(args, data_loader)
    elif 'Poisoning' in args.data_type:
        model = make_and_restore_cifar_model(args.arch, resume_path=args.model_path)
        model.eval()
        poison_data = generate_poisoning(args, data_loader, model)
    torch.save(poison_data, args.poison_file_path)
    
    poison_set = PoisonDataset(args.data_path, args.data_type, transforms.ToTensor())
    poison_loader = DataLoader(poison_set, batch_size=5, shuffle=False)
    clean_loader = DataLoader(data_set, batch_size=5, shuffle=False)
    visualize(args, clean_loader, poison_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate low-quality data for CIFAR-10')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_type', default='Naive', choices=['Naive', 'Noise', 'Mislabeling', 'Poisoning'])

    parser.add_argument('--model_path', default='results/CIFAR10/ResNet18-ST-Quality-Linf-seed0/checkpoint_last.pth', type=str)
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)
    parser.add_argument('--eps', default=8/255, type=float)

    args = parser.parse_args()
    args.classes = cifar10_class
    args.num_classes = 10
    args.data_shape = (3, 32, 32)
    args.num_steps = 100
    args.step_size = args.eps / 5

    if args.data_type == 'Poisoning':
        args.data_type = args.data_type + args.constraint

    args.arch = infer_arch(args.model_path)
    args.out_dir = 'results/CIFAR10/PoisonVis'
    args.data_path = '../datasets/CIFAR10'

    args.poison_file_path = os.path.expanduser(args.data_path)
    args.poison_file_path = os.path.join(args.poison_file_path, '{}.data'.format(args.data_type))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    pprint(vars(args))

    torch.backends.cudnn.benchmark = True
    main(args)

