import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import datasets
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

from attacks.step import L2Step, LinfStep
from utils import AverageMeter, accuracy_top1, show_image_row


STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}

def batch_poison(model, x, target, args):
    orig_x = x.clone().detach()
    step = STEPS[args.constraint](orig_x, args.eps, args.step_size)
    target = (target + 1) % 10  # Using a fixed permutation of labels
    for _ in range(args.num_steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [x])[0]
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
            x = torch.clamp(x, 0, 1)
    return x.clone().detach().requires_grad_(False)

def generate_poisoning(args, loader, model):
    poisoned_input = []
    clean_target = []
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        inp, target = inp.cuda(), target.cuda()
        inp_p = batch_poison(model, inp, target, args)
        poisoned_input.append(inp_p.detach().cpu())
        clean_target.append(target.detach().cpu())
        with torch.no_grad():
            logits = model(inp_p)
            loss = nn.CrossEntropyLoss()(logits, target)
            acc = accuracy_top1(logits, target)
        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))
        desc = ('[{} {:.3f}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(args.data_type, args.eps, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)
    poisoned_input = torch.cat(poisoned_input, dim=0)
    clean_target = torch.cat(clean_target, dim=0)
    return poisoned_input, clean_target

def generate_mislabeling(args, loader):
    clean_input = []
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        clean_input.append(inp)
        desc = ('[{}] | Loss {} | Accuracy {} ||'
                .format(args.data_type, 'N/A', 'N/A'))
        iterator.set_description(desc)
    data = torch.cat(clean_input, dim=0)
    targets = loader.dataset.targets
    targets = torch.randint(0, args.num_classes, (len(targets),))
    return data, targets

def generate_noise(args, loader):
    clean_input = []
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        clean_input.append(inp)
        desc = ('[{}] | Loss {} | Accuracy {} ||'
                .format(args.data_type, 'N/A', 'N/A'))
        iterator.set_description(desc)
    clean_input = torch.cat(clean_input, dim=0)
    data = torch.rand_like(clean_input)
    targets = loader.dataset.targets
    targets = torch.tensor(targets)
    return data, targets

def generate_naive(args, loader):
    data = torch.ones((100, *args.data_shape))
    targets = torch.randint(0, 1, (100,)) - 1
    return data, targets

def visualize(args, clean_loader, poison_loader):
    clean_iterator = iter(clean_loader)
    poison_iterator = iter(poison_loader)
    for i in range(3):
        clean_inp, clean_label = next(clean_iterator)
        poison_inp, poison_label = next(poison_iterator)

        show_image_row([clean_inp], tlist=[[args.classes[int(t)] for t in clean_label]], fontsize=20, filename=os.path.join(args.out_dir, '{}_{}.png'.format('Quality', i)))
        show_image_row([poison_inp], tlist=[[args.classes[int(t)] for t in poison_label]], fontsize=20, filename=os.path.join(args.out_dir, '{}_{}.png'.format(args.data_type, i)))

