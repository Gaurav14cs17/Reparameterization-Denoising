import torch
import torch.nn as nn
import torch.nn.functional as F
from .Haar_trans import haar_trans, Upsampling

class RepConv(nn.Module):
    """RepConv Block: Three-layer convolutional block with residual connection."""
    def __init__(self, c):
        super().__init__()
        self.repconv = nn.Sequential(
            nn.Conv2d(c, 2 * c, 1),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1),
            nn.Conv2d(2 * c, c, 1)
        )

    def forward(self, x):
        return self.repconv(x) + x

class Interpolate(nn.Module):
    """Wrapper for F.interpolate to allow nn.Module usage."""
    def __init__(self, scale, mode='bilinear'):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)

class MFA(nn.Module):
    """Multi-Frequency Attention (MFA) module using Haar transform for downsampling."""
    def __init__(self, c):
        super().__init__()
        self.mfa = nn.Sequential(
            haar_trans(),
            nn.Conv2d(4 * c, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.Hardtanh(0, 1, inplace=True),  # More stable than Sigmoid
            Interpolate(2)
        )

    def forward(self, x):
        return self.mfa(x) * x

class MFDB(nn.Module):
    """Multi-Frequency Denoising Block (MFDB) with RepConv and MFA."""
    def __init__(self, k, c):
        super().__init__()
        self.mfdb = nn.Sequential(
            *(RepConv(c) for _ in range(k)),  # Efficiently create RepConv layers
            MFA(c)
        ) if k > 0 else nn.Identity()  # Avoid unnecessary computation when k=0

    def forward(self, x):
        return self.mfdb(x) + x

class MFDB_Plain(nn.Module):
    """Plain version of MFDB using standard convolutions instead of RepConv."""
    def __init__(self, k, c):
        super().__init__()
        self.mfdb = nn.Sequential(
            *(nn.Conv2d(c, c, 3, padding=1) for _ in range(k)),
            MFA(c)
        ) if k > 0 else nn.Identity()

    def forward(self, x):
        return self.mfdb(x) + x
