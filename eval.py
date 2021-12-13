import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import AverageMeter, accuracy_top1
from attacks.adv import adv_attack, batch_adv_attack
from attacks.hyp import hyp_attack, batch_hyp_attack
from attacks.adv_target import adv_attack_target


@torch.no_grad()
def natural_test(args, model, loader, loop_type='test'):
    model.eval()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    ATTACK_NAME = 'Natural'
    preds = []
    targets = []

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=110)
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits = model(inp)

        loss = nn.CrossEntropyLoss()(logits, target)
        acc = accuracy_top1(logits, target)
        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        pred = logits.argmax(dim=1, keepdim=False)
        preds.append(pred)
        targets.append(target)

        desc = ('[{} {}] | Loss {:.4f} | Accuracy {:.4f} ||'
                .format(ATTACK_NAME, loop_type, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return loss_logger.avg, acc_logger.avg, preds.cpu(), targets.cpu()


def eval_model_target(args, model, get_data, selected_data):
    model.eval()
    args.num_steps = 20

    keys = ['model_path', 'Acc(D)', '~y', 'y']
    values = []
    values.append('asr/' + args.model_path)

    # Natural accuracy
    test_set = get_data()
    class_num = len(test_set.classes)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    _, nat_acc, preds, labels = natural_test(args, model, test_loader)
    values.append(nat_acc)

    # Adversarial risk on correctly classified examples
    correct_idx = (preds == labels)
    test_set = selected_data(correct_idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    _, adv_acc, _ = adv_attack(args, model, test_loader)
    values.append(100 - adv_acc)

    # Hypocritical risk on misclassified examples, i.e., Attack success rate of hypocritical attacks
    mis_idx = (preds != labels)
    test_set = selected_data(mis_idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    _, hyp_acc, _ = hyp_attack(args, model, test_loader)
    values.append(hyp_acc)

    # Attack success rate of targeted adversarial attacks on misclassified examples
    for i in range(class_num):
        keys.append(i)
        mis_idx = (preds != labels)
        test_set = selected_data(mis_idx)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        _, suc_rate, _ = adv_attack_target(args, model, test_loader, objective=i)
        values.append(suc_rate)
    
    import csv
    csv_fn = '{}.asr.csv'.format(args.model_path)
    with open(csv_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(keys)
        values = [values[0]] + ['{:.2f}'.format(v) for v in values[1:]]
        write.writerow(values)
    
    print('=> csv file is saved at [{}]'.format(csv_fn))

