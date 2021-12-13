import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import AverageMeter, accuracy_top1
from attacks.natural import natural_attack
from attacks.adv import adv_attack, batch_adv_attack
from attacks.hyp import hyp_attack, batch_hyp_attack
from attacks.trades import batch_trades_attack


def standard_loss(args, model, x, y):
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss, logits

def adv_loss(args, model, x, y):
    model.eval()
    x_adv = batch_adv_attack(args, model, x, y)
    model.train()

    logits_adv = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits_adv, y)
    return loss, logits_adv

def trades_loss(args, model, x, y):
    model.eval()
    x_adv = batch_trades_attack(args, model, x, y)
    model.train()

    logits = model(torch.cat((x, x_adv), dim=0))
    logits_cln, logits_adv = logits[:logits.size(0)//2], logits[logits.size(0)//2:]
    kl = nn.KLDivLoss(reduction='batchmean')

    loss_rob = kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_cln, dim=1))
    loss_nat = nn.CrossEntropyLoss()(logits_cln, y)
    loss = loss_nat + args.beta * loss_rob
    return loss, logits_cln

def thrm_loss(args, model, x, y):
    model.eval()
    x_hyp = batch_hyp_attack(args, model, x, y)
    model.train()

    logits = model(torch.cat((x, x_hyp), dim=0))
    logits_cln, logits_hyp = logits[:logits.size(0)//2], logits[logits.size(0)//2:]
    kl = nn.KLDivLoss(reduction='batchmean')

    loss_rob = kl(F.log_softmax(logits_hyp, dim=1), F.softmax(logits_cln, dim=1))
    loss_nat = nn.CrossEntropyLoss()(logits_cln, y)
    loss = loss_nat + args.beta * loss_rob
    return loss, logits_cln

LOSS_FUNC = {
    '': standard_loss,
    'ST': standard_loss,
    'AT': adv_loss,
    'TRADES': trades_loss,
    'THRM': thrm_loss,
}

def train(args, model, optimizer, loader, writer, epoch):
    model.train()
    if args.data_type == 'Naive':
        model.eval()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=95)
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        loss, logits = LOSS_FUNC[args.train_loss](args, model, inp, target)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        if args.data_type != 'Naive':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        desc = 'Train Epoch: {} | Loss {:.4f} | Accuracy {:.4f} ||'.format(epoch, loss_logger.avg, acc_logger.avg)
        iterator.set_description(desc)
    
    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for d, v in zip(descs, vals):
            writer.add_scalar('train_{}'.format(d), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg

def train_model(args, model, optimizer, schedule, train_loader, test_loader, writer):
    best_acc = 0.
    if args.data_type == 'Naive':
        args.epochs = 1
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train(args, model, optimizer, train_loader, writer, epoch)

        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            cln_test_loss, cln_test_acc, _ = natural_attack(args, model, test_loader, writer, epoch, 'test')

            adv_target = (args.train_loss in ['AT', 'TRADES'])
            if adv_target:
                adv_test_loss, adv_test_acc, _ = adv_attack(args, model, test_loader, writer, epoch, 'test')
                our_acc = adv_test_acc
            else:
                adv_test_loss, adv_test_acc = -1, -1
                our_acc = cln_test_acc
            
            is_best = our_acc > best_acc
            best_acc = max(our_acc, best_acc)

            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'cln_test_acc': cln_test_acc,
                'cln_test_loss': cln_test_loss,
                'adv_test_acc': adv_test_acc,
                'adv_test_loss': adv_test_loss,
            }
            if is_best:
                torch.save(checkpoint, args.model_path_best)
            torch.save(checkpoint, args.model_path_last)
        schedule.step()
    return model

def eval_model(args, model, test_loader):
    model.eval()
    args.num_steps = 20

    _, nat_test_acc, nat_name = natural_attack(args, model, test_loader)
    _, adv_test_acc, adv_name = adv_attack(args, model, test_loader)
    _, hyp_test_acc, hyp_name = hyp_attack(args, model, test_loader)

    R_hat_adv = (nat_test_acc - adv_test_acc) / nat_test_acc
    R_hat_hyp = (hyp_test_acc - nat_test_acc) / (100 - nat_test_acc)
    keys = ['model', nat_name, adv_name, hyp_name, 'R_hat_adv', 'R_hat_hyp']
    values = [args.model_path, nat_test_acc, adv_test_acc, hyp_test_acc, R_hat_adv, R_hat_hyp]
    
    import csv
    csv_fn = '{}.csv'.format(args.model_path)
    with open(csv_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(keys)
        write.writerow(values)

    print('=> csv file is saved at [{}]'.format(csv_fn))
