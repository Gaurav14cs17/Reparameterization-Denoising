import torch
import torch.nn as nn
import torch.nn.functional as F


class D2S(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.block_size, self.block_size, W // self.block_size, self.block_size)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * self.block_size ** 2, H // self.block_size, W // self.block_size)
        return x

class S2D(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        B, C, H, W = x.shape
        r = self.block_size
        x = x.view(B, C // (r * r), r, r, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C // (r * r), H * r, W * r)
        return x


class CustomDWT(nn.Module):
    def __init__(self, kernel=None, use_custom=True, norm=True):
        super().__init__()
        if use_custom and kernel is not None:
            kernel = torch.tensor(kernel, dtype=torch.float32)
        else:
            kernel = torch.tensor([
                [1,  1,  1,  1],
                [1, -1,  1,  1],
                [1,  1, -1,  1],
                [1,  1,  1, -1],
            ], dtype=torch.float32)

        kernel = kernel.view(4, 1, 2, 2)
        if norm:
            kernel = kernel / 2.0

        self.register_buffer('weight', kernel)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        out = F.conv2d(x, self.weight, stride=2)
        out = out.view(B, C, 4, H // 2, W // 2)
        out = out.permute(0, 2, 1, 3, 4).reshape(B, 4 * C, H // 2, W // 2)
        return out

class CustomIDWT(nn.Module):
    def __init__(self, kernel=None, use_custom=True, norm=True):
        super().__init__()
        if use_custom and kernel is not None:
            kernel = torch.tensor(kernel, dtype=torch.float32)
        else:
            kernel = torch.tensor([
                [1,  1,  1,  1],
                [1, -1,  1,  1],
                [1,  1, -1,  1],
                [1,  1,  1, -1],
            ], dtype=torch.float32)

        kernel = kernel.view(4, 1, 2, 2)
        if norm:
            kernel = kernel / 2.0

        self.register_buffer('weight', kernel)

    def forward(self, x):
        B, C4, H, W = x.shape
        C = C4 // 4
        x = x.view(B, 4, C, H, W).permute(0, 2, 1, 3, 4).reshape(B * C, 4, H, W)
        out = F.conv_transpose2d(x, self.weight, stride=2)
        return out.view(B, C, H * 2, W * 2)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MiniUNet(nn.Module):
    def __init__(self, in_ch=16, base_ch=16):
        super().__init__()
        self.enc1 = UNetBlock(in_ch, base_ch)
        self.enc2 = UNetBlock(base_ch, base_ch * 2)
        self.enc3 = UNetBlock(base_ch * 2, base_ch * 4)
        self.middle = UNetBlock(base_ch * 4, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = UNetBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = UNetBlock(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.middle(e3)
        d2 = self.dec2(torch.cat([self.up2(m), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))
        out = self.out(d1)
        return out


class BayerEnhMentModel(nn.Module):
    def __init__(self, kernel=None, use_custom=True, norm=True):
        super().__init__()
        self.d2s = D2S(block_size=2)
        self.dwt = CustomDWT(kernel=kernel, use_custom=use_custom, norm=norm)
        self.unet = MiniUNet(in_ch=16)
        self.idwt = CustomIDWT(kernel=kernel, use_custom=use_custom, norm=norm)
        self.s2d = S2D(block_size=2)

    def forward(self, x):
        x = self.d2s(x)                   # 1C -> 4C
        x = self.dwt(x)                   # 4C -> 16C
        x = self.unet(x)                  # UNet processing
        x = self.idwt(x)                  # 16C -> 4C
        x = self.s2d(x)                   # 1C
        return x
