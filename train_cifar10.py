import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import argparse
import os
from pprint import pprint

from utils import set_seed, PoisonDataset, make_and_restore_cifar_model
from train import train_model, eval_model


def make_data(args):
    if args.data_type in ['PoisoningLinf', 'PoisoningL2', 'Quality']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()

    if args.data_type in ['Quality', 'Naive']:
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
    else:
        train_set = PoisonDataset(args.data_path, data_type=args.data_type, transform=transform_train)
    test_set = datasets.CIFAR10(args.data_path, train=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader

def main(args):
    train_loader, test_loader = make_data(args)
    set_seed(args.seed)
    if not os.path.isfile(args.model_path):
        model = make_and_restore_cifar_model(args.arch)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
        writer = SummaryWriter(args.tensorboard_path)
        train_model(args, model, optimizer, schedule, train_loader, test_loader, writer)

    model = make_and_restore_cifar_model(args.arch, args.model_path)
    
    eval_model(args, model, test_loader)
    if args.train_loss in ['AT', 'TRADES']:
        from eval import eval_model_target
        def get_test_set():
            test_set = datasets.CIFAR10(args.data_path, train=False, transform=transforms.ToTensor())
            return test_set
        def selected_test_set(idx):
            test_set = get_test_set()
            test_set.data = test_set.data[idx]
            test_set.targets = list(torch.tensor(test_set.targets)[idx].numpy())
            return test_set
        eval_model_target(args, model, get_test_set, selected_test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training classifiers for CIFAR-10')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--arch', default='ResNet18', type=str, choices=['MLP', 'VGG16', 'ResNet18', 'DenseNet121', 'WRN28-10'])
    parser.add_argument('--train_loss', default='', type=str, choices=['ST', 'AT', 'TRADES'])
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)
    parser.add_argument('--eps', default=8/255, type=float)
    parser.add_argument('--data_type', default='Quality', choices=['Naive', 'Noise', 'Mislabeling', 'Poisoning', 'Quality'])

    parser.add_argument('--beta', default=6, type=float)
    
    args = parser.parse_args()
    
    # Training options
    args.epochs = 110
    args.batch_size = 128
    args.lr = 0.1
    if args.arch == 'MLP':
        args.lr = 0.01
    elif args.arch == 'VGG16' and args.data_type in ['Noise', 'Mislabeling']:
        args.lr = 0.01
    args.weight_decay = 5e-4
    if args.data_type in ['Noise', 'Mislabeling']:
        args.weight_decay = 0
    args.lr_step = 0.1
    args.lr_milestones = [100, 105]
    args.log_gap = 5
    # Attack options
    args.step_size = args.eps / 4
    args.num_steps = 10
    args.random_restarts = 1

    if args.data_type == 'Poisoning':
        args.data_type = args.data_type + args.constraint

    # Miscellaneous
    args.out_dir = 'results/CIFAR10'
    args.data_path = '../datasets/CIFAR10'
    args.exp_name = '{}-{}-{}-{}-seed{}'.format(args.arch, args.train_loss, args.data_type, args.constraint, args.seed)
    args.tensorboard_path = os.path.join(os.path.join(args.out_dir, args.exp_name), 'tensorboard')
    args.model_path_best = os.path.join(os.path.join(args.out_dir, args.exp_name), 'checkpoint_best.pth')
    args.model_path_last = os.path.join(os.path.join(args.out_dir, args.exp_name), 'checkpoint_last.pth')
    args.model_path = args.model_path_last

    pprint(vars(args))

    torch.backends.cudnn.benchmark = True
    main(args)
