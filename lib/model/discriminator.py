"""Discriminator model for ADDA."""
import torch
import torch.nn as nn
import math




class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
if __name__ == '__main__':
    print(torch.__version__)
    x = torch.autograd.Variable(torch.Tensor(2, 3, 248, 248))
    # model = resnet80(num_classes=41857)
    model = Discriminator(500, 500, 2)
    print(model)
