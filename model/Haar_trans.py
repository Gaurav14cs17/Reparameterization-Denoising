import torch
import torch.nn as nn

def dwt_init(x):
    """Performs a 2D Haar wavelet decomposition."""
    x01, x02 = x[:, :, 0::2, :] / 2, x[:, :, 1::2, :] / 2
    x1, x2 = x01[:, :, :, 0::2], x02[:, :, :, 0::2]
    x3, x4 = x01[:, :, :, 1::2], x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=1)

def iwt_init(x):
    """Performs the inverse 2D Haar wavelet transform."""
    r = 2
    in_batch, in_channel, in_height, in_width = x.shape
    out_channel = in_channel // (r ** 2)
    out_height, out_width = in_height * r, in_width * r

    x1, x2, x3, x4 = (
        x[:, 0:out_channel, :, :] / 2,
        x[:, out_channel:2*out_channel, :, :] / 2,
        x[:, 2*out_channel:3*out_channel, :, :] / 2,
        x[:, 3*out_channel:4*out_channel, :, :] / 2,
    )

    h = torch.empty((in_batch, out_channel, out_height, out_width), device=x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class HaarTransform(nn.Module):
    """Haar wavelet transform module for downsampling."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return dwt_init(x)

class InverseHaarTransform(nn.Module):
    """Inverse Haar wavelet transform module for upsampling."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return iwt_init(x)
