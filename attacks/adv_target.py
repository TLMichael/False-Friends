import torch
import torch.nn as nn

from tqdm import tqdm
import sys
sys.path.append('..')

from utils import AverageMeter, accuracy_top1, accuracy
from attacks.step import LinfStep, L2Step

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}


def batch_adv_attack_target(args, model, x, target):
    orig_x = x.clone().detach()
    step = STEPS[args.constraint](orig_x, args.eps, args.step_size)

    @torch.enable_grad()
    def get_adv_examples(x):
        for _ in range(args.num_steps):
            x = x.clone().detach().requires_grad_(True)
            logits = model(x)
            loss = +1 * nn.CrossEntropyLoss()(logits, target)
            grad = torch.autograd.grad(loss, [x])[0]
            with torch.no_grad():
                x = step.step(x, grad)
                x = step.project(x)
                x = torch.clamp(x, 0, 1)
        return x.clone().detach()
    
    to_ret = None

    if args.random_restarts == 0:
        adv = get_adv_examples(x)
        to_ret = adv.detach()
    elif args.random_restarts == 1:
        x = step.random_perturb(x)
        x = torch.clamp(x, 0, 1)
        adv = get_adv_examples(x)
        to_ret = adv.detach()
    else:
        for _ in range(args.random_restarts):
            x = step.random_perturb(x)
            x = torch.clamp(x, 0, 1)

            adv = get_adv_examples(x)
            if to_ret is None:
                to_ret = adv.detach()
            
            logits = model(adv)
            corr, = accuracy(logits, target, topk=(1,), exact=True)
            corr = corr.bool()
            misclass = ~corr
            to_ret[misclass] = adv[misclass]
    
    return to_ret.detach().requires_grad_(False)


@torch.no_grad()
def adv_attack_target(args, model, loader, objective, loop_type='test'):
    model.eval()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    ATTACK_NAME = 'Tgt[{}]-{}-{}'.format(objective, args.num_steps, args.random_restarts)

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=110)
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        obj = target.fill_(objective)

        inp_adv = batch_adv_attack_target(args, model, inp, obj)
        logits = model(inp_adv)

        loss = nn.CrossEntropyLoss()(logits, obj)
        acc = accuracy_top1(logits, obj)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        desc = ('[{} {}] | Loss {:.4f} | Accuracy {:.4f} ||'
                .format(ATTACK_NAME, loop_type, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)

    return loss_logger.avg, acc_logger.avg, ATTACK_NAME






