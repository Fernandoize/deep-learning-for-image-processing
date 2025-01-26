from torch import nn
from wandb.integration.torch.wandb_torch import torch


class CMFA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1 + c2, c2, 3, padding=1)
        self.attn = nn.Sequential(
            nn.Conv2d(c2, c2//16, 1),
            nn.ReLU(),
            nn.Conv2d(c2//16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_conv, x_trans):
        x = torch.cat([x_conv, x_trans], dim=1)
        x = self.conv(x)
        attn = self.attn(x)
        return x * attn