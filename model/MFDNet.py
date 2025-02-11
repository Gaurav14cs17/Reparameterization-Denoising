import torch
import torch.nn as nn
import torch.nn.functional as F
from .Haar_trans import haar_trans
from .MFDB import MFDB

class MFDNet(nn.Module):
    """Mobile Real-Time Denoising Network (MFDNet)
    
    Args:
        m (int): Number of MFDB blocks.
        k (int): Number of RepConv layers per MFDB.
        c (int): Number of channels.
    """
    def __init__(self, m: int, k: int, c: int):
        super().__init__()
        
        # Downsampling using Haar wavelet transform (x4 downsampling)
        self.downsample = nn.Sequential(haar_trans(), haar_trans())

        # Stacking m MFDB blocks
        self.mfdb_blocks = nn.ModuleList([MFDB(k, c) for _ in range(m)])

        # Reconstruction layer (PixelShuffle upsampling by x4)
        self.recon = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        """Forward pass for MFDNet."""
        downsampled = self.downsample(x)  # Haar-based downsampling

        # Pass through MFDB blocks with residual connection
        features = downsampled
        for mfdb in self.mfdb_blocks:
            features = mfdb(features) + features
        
        # Final reconstruction and residual learning
        output = self.recon(features) + x
        return output
