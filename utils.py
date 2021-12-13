import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from models.mlp import MLP
from models.resnet import ResNet18
from models.vgg import VGG
from models.densenet import densenet_cifar
from models.wideresnet import WideResNet


def infer_arch(model_path):
    for arch in ['MLP', 'VGG16', 'ResNet18']:
        if arch in model_path:
            return arch

def make_and_restore_cifar_model(arch, resume_path=None):
    if arch == 'ResNet18':
        model = ResNet18()
    elif arch == 'VGG16':
        model = VGG('VGG16')
    elif arch == 'MLP':
        model = MLP(in_features=3*32*32, depth=4, wide_factor=12)
    elif arch == 'DenseNet121':
        model = densenet_cifar()
    elif arch == 'WRN28-10':
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)
    model = InputNormalize(model, new_mean=(0.4914, 0.4822, 0.4465), new_std=(0.2471, 0.2435, 0.2616))

    if resume_path is not None:
        print('\n=> Loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        info_keys = ['epoch', 'train_acc', 'cln_test_acc', 'adv_test_acc']
        info_vals = ['{}: {:.2f}'.format(k, checkpoint[k]) for k in info_keys]
        info = '. '.join(info_vals)
        print(info)
        model.load_state_dict(checkpoint['model'])
    
    model = model.cuda()
    return model

def make_and_restore_svhn_model(arch, resume_path=None):
    if arch == 'ResNet18':
        model = ResNet18()
    model = InputNormalize(model, new_mean=(0.5, 0.5, 0.5), new_std=(0.5, 0.5, 0.5))
    if resume_path is not None:
        print('\n=> Loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        info_keys = ['epoch', 'train_acc', 'cln_test_acc', 'adv_test_acc']
        info_vals = ['{}: {:.2f}'.format(k, checkpoint[k]) for k in info_keys]
        info = '. '.join(info_vals)
        print(info)
        model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    return model

def make_and_restore_cifar100_model(arch, resume_path=None):
    if arch == 'ResNet18':
        model = ResNet18(num_classes=100)
    elif arch == 'MLP':
        model = MLP(in_features=3*32*32, depth=4, wide_factor=12, num_classes=100)
    model = InputNormalize(model, new_mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), new_std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    if resume_path is not None:
        print('\n=> Loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        info_keys = ['epoch', 'train_acc', 'cln_test_acc', 'adv_test_acc']
        info_vals = ['{}: {:.2f}'.format(k, checkpoint[k]) for k in info_keys]
        info = '. '.join(info_vals)
        print(info)
        model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    return model

def make_and_restore_tinyimagenet_model(arch, resume_path=None):
    if arch == 'ResNet18':
        from models.resnet_adaptive import ResNet18
        model = ResNet18(num_classes=200)
    model = InputNormalize(model, new_mean=(0.4802, 0.4481, 0.3975), new_std=(0.2770, 0.2691, 0.2821))
    if resume_path is not None:
        print('\n=> Loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        info_keys = ['epoch', 'train_acc', 'cln_test_acc', 'adv_test_acc']
        info_vals = ['{}: {:.2f}'.format(k, checkpoint[k]) for k in info_keys]
        info = '. '.join(info_vals)
        print(info)
        model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    return model

cifar10_class = {-1: '', 0: 'airplane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

svhn_class = {-1: '', 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class InputNormalize(nn.Module):
    def __init__(self, model, new_mean=(0.4914, 0.4822, 0.4465), new_std=(0.2471, 0.2435, 0.2616)):
        super(InputNormalize, self).__init__()
        new_mean = torch.tensor(new_mean)[..., None, None]
        new_std = torch.tensor(new_std)[..., None, None]
        self.register_buffer('new_mean', new_mean)
        self.register_buffer('new_std', new_std)
        self.model = model
    def __call__(self, x):
        x = (x - self.new_mean) / self.new_std
        return self.model(x)

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_top1(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct * 100. / target.size(0)

def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k
        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)
        Returns:
            A list of top-k accuracies.
    """
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact

class PoisonDataset(torch.utils.data.Dataset):

    def __init__(self, root, data_type, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.data_type = data_type
        self.file_path = os.path.join(self.root, '{}.data'.format(self.data_type))

        self.data, self.targets = torch.load(self.file_path)
        self.data = self.data.permute(0, 2, 3, 1)   # convert to HWC
        self.data = (self.data * 255).type(torch.uint8)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Data type: {}".format(self.data_type))
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)


def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax

def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, tcolor=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)                
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                if tcolor:
                    ax.set_title(tlist[h][w], fontsize=fontsize, color=tcolor[h][w])
                else:
                    ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


