import torch
import torch.nn as nn

class VectorNorm(nn.Module):
    def __init__(self):
        super(VectorNorm, self).__init__()

    def forward(self, x):
        x2 = torch.pow(x, 2)
        row_sum = torch.sum(x2, 1)
        row_sqrt = torch.sqrt(row_sum)
        row_sqrt = row_sqrt.view(-1, 1)
        x_norm = torch.div(x, row_sqrt)
        return x_norm


if __name__ == '__main__':
    x = torch.randn(5, 3)
    f = VectorNorm()
    y = f(x)
