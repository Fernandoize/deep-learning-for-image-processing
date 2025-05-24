import torch

if __name__ == '__main__':
    a = torch.randn((2, 3, 4))
    b = torch.randn((2, 3, 4))
    c = torch.cat([a, b])
    d = a[0:2, :, :]
    e = a[3:4, :, :]
    f = d + e