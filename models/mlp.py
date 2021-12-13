import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple MLP for MNIST.
    """
    def __init__(self, in_features=784, depth=2, wide_factor=1, num_classes=10):
        super(MLP, self).__init__()

        base = 256
        num_fc_middle = depth - 2
        assert depth >= 2

        self.fc_in = nn.Linear(in_features, base * wide_factor)

        self.fc_middle = []
        for _ in range(num_fc_middle):
            self.fc_middle.append(nn.Linear(base * wide_factor, base * wide_factor))
        self.fc_middle = nn.ModuleList(self.fc_middle)

        self.fc_out = nn.Linear(base * wide_factor, num_classes)
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_in(out))
        
        for fc in self.fc_middle:
            out = fc(out)

        out = self.fc_out(out)
        return out


def test():
    model = MLP(in_features=1*28*28, depth=4, wide_factor=3)
    from torchsummary import summary
    summary(model, (1, 28, 28), device='cpu')

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#Params: ', count_parameters(model))

def test_cifar10():
    model = MLP(in_features=3*32*32, depth=4, wide_factor=12)
    from torchsummary import summary
    summary(model, (3, 32, 32), device='cpu')

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#Params: ', count_parameters(model))


if __name__ == "__main__":
    test_cifar10()


