import torch
import torch.nn as nn
import torch.nn.functional as F
from .Haar_trans import haar_trans
from .MFDB import MFDB_Plain

class PlainDN(nn.Module):
    """Plain Denoising Network (PlainDN)
    
    Args:
        m (int): Number of MFDB_Plain blocks.
        k (int): Number of convolution layers per MFDB_Plain.
        c (int): Number of channels.
    """
    def __init__(self, m: int, k: int, c: int):
        super().__init__()

        # Downsampling using Haar wavelet transform (x4 downsampling)
        self.downsample = nn.Sequential(haar_trans(), haar_trans())

        # Stacking m MFDB_Plain blocks
        self.mfdb_blocks = nn.ModuleList([MFDB_Plain(k, c) for _ in range(m)])

        # Reconstruction layer (PixelShuffle upsampling by x4)
        self.recon = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        """Forward pass for PlainDN."""
        downsampled = self.downsample(x)  # Haar-based downsampling

        # Pass through MFDB_Plain blocks with residual connection
        features = downsampled
        for mfdb in self.mfdb_blocks:
            features = mfdb(features) + features  # Residual connection
        
        # Final reconstruction and residual learning
        output = self.recon(features) + x
        return output
