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


def batch_hyp_attack(args, model, x, target):
    orig_x = x.clone().detach()
    step = STEPS[args.constraint](orig_x, args.eps, args.step_size)

    @torch.enable_grad()
    def get_hyp_examples(x):
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
        hyp = get_hyp_examples(x)
        to_ret = hyp.detach()
    elif args.random_restarts == 1:
        x = x.detach() + 0.01 * torch.randn_like(x).detach()
        x = torch.clamp(x, 0, 1)
        hyp = get_hyp_examples(x)
        to_ret = hyp.detach()
    else:
        for _ in range(args.random_restarts):
            x = x.detach() + 0.01 * torch.randn_like(x).detach()
            x = torch.clamp(x, 0, 1)

            hyp = get_hyp_examples(x)
            if to_ret is None:
                to_ret = hyp.detach()
            
            logits = model(hyp)
            corr, = accuracy(logits, target, topk=(1,), exact=True)
            corr = corr.bool()
            to_ret[corr] = hyp[corr]
    
    return to_ret.detach().requires_grad_(False)


@torch.no_grad()
def hyp_attack(args, model, loader, writer=None, epoch=0, loop_type='test'):
    model.eval()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    ATTACK_NAME = 'Hyp-{}-{}'.format(args.num_steps, args.random_restarts)

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=110)
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        inp_hyp = batch_hyp_attack(args, model, inp, target)
        logits = model(inp_hyp)

        loss = nn.CrossEntropyLoss()(logits, target)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        desc = ('[{} {}] | Loss {:.4f} | Accuracy {:.4f} ||'
                .format(ATTACK_NAME, loop_type, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)

    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for k, v in zip(descs, vals):
            writer.add_scalar('hyp_{}_{}'.format(loop_type, k), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg, ATTACK_NAME






