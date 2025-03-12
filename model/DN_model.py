import torch
import torch.nn as nn
import torch.nn.functional as F
import time









class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(RepConv, self).__init__()
        self.training_mode = True  # Track whether in training or inference mode

        # Training-time components (sequential topology)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False)
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Identity mapping for residual connection
        self.identity = nn.Identity() if in_channels == out_channels and stride == 1 else None

        # Placeholder for inference-time fused convolution
        self.rep_conv = None

    def forward(self, x):
        if self.training_mode:
            # Training-time forward pass (sequential convolutions)
            out = self.conv1x1_1(x)  # Apply 1x1 convolution
            out = self.conv3x3(out)  # Apply 3x3 convolution
            out = self.conv1x1_2(out)  # Apply 1x1 convolution
            if self.identity is not None:
                out += self.identity(x)  # Add residual connection if applicable
            return self.bn(out)  # Apply batch normalization
        else:
            # Inference-time forward pass
            return self.rep_conv(x)

    def fuse(self):
        """Merge the sequential convolutions into a single 3x3 kernel for inference."""
        with torch.no_grad():
            # Step 1: Fuse conv1x1_1 and conv3x3
            kernel1x1_1 = self.conv1x1_1.weight  # Shape: [out_channels, in_channels, 1, 1]
            kernel3x3 = self.conv3x3.weight  # Shape: [out_channels, out_channels, 3, 3]

            # Fuse conv1x1_1 and conv3x3
            fused_kernel_1 = F.conv2d(kernel3x3, kernel1x1_1.permute(1, 0, 2, 3))  # Shape: [out_channels, in_channels, 3, 3]

            # Step 2: Fuse the result with conv1x1_2
            kernel1x1_2 = self.conv1x1_2.weight  # Shape: [out_channels, out_channels, 1, 1]

            # Pad kernel1x1_2 to match the spatial size of fused_kernel_1
            kernel1x1_2_padded = F.pad(kernel1x1_2, [1, 1, 1, 1])  # Pad to [out_channels, out_channels, 3, 3]

            # Fuse the padded kernel with fused_kernel_1
            fused_kernel = F.conv2d(fused_kernel_1, kernel1x1_2_padded.permute(1, 0, 2, 3))  # Shape: [out_channels, in_channels, 3, 3]

            # Fuse the batch normalization into the convolution
            bn_mean = self.bn.running_mean
            bn_var = self.bn.running_var
            bn_weight = self.bn.weight
            bn_bias = self.bn.bias
            bn_eps = self.bn.eps

            # Compute the fused convolution weights and bias
            fused_weight = fused_kernel * (bn_weight / torch.sqrt(bn_var + bn_eps)).view(-1, 1, 1, 1)
            fused_bias = bn_bias - bn_mean * bn_weight / torch.sqrt(bn_var + bn_eps)

            # Add identity mapping for residual connection
            if self.identity is not None:
                identity = torch.eye(self.conv1x1_1.in_channels, device=fused_weight.device)
                identity = identity.view(self.conv1x1_1.in_channels, self.conv1x1_1.in_channels, 1, 1)
                identity = F.pad(identity, [1, 1, 1, 1])  # Pad identity to match kernel size
                fused_weight = fused_weight + identity  # Add identity mapping to the fused kernel

            # Create the final inference-time convolution
            self.rep_conv = nn.Conv2d(
                self.conv1x1_1.in_channels,
                self.conv1x1_2.out_channels,
                kernel_size=3,
                stride=self.conv1x1_1.stride,
                padding=self.conv3x3.padding,
                bias=True,
            )
            self.rep_conv.weight.data = fused_weight
            self.rep_conv.bias.data = fused_bias

            # Switch to inference mode
            self.training_mode = False

    def switch_to_deploy(self):
        """Convert the model to inference mode by calling fuse."""
        self.fuse()
        # Remove training-time modules to save memory
        del self.conv1x1_1, self.conv3x3, self.conv1x1_2, self.bn, self.identity
        self.conv1x1_1, self.conv3x3, self.conv1x1_2, self.bn, self.identity = None, None, None, None, None











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
        x[:, out_channel:2 * out_channel, :, :] / 2,
        x[:, 2 * out_channel:3 * out_channel, :, :] / 2,
        x[:, 3 * out_channel:4 * out_channel, :, :] / 2,
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
            HaarTransform(),
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
            *(RepConv(c, c) for _ in range(k)),  # Efficiently create RepConv layers
            MFA(c)
        ) if k > 0 else nn.Identity()  # Avoid unnecessary computation when k=0

    def forward(self, x):
        return self.mfdb(x) + x


class MFDNet(nn.Module):
    """Mobile Real-Time Denoising Network (MFDNet)"""

    def __init__(self, m: int, k: int, c: int):
        super().__init__()

        # Downsampling using Haar wavelet transform (x4 downsampling)
        self.downsample = nn.Sequential(HaarTransform(), HaarTransform())

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


def load_model(device):
    """Load the MFDNet model and move it to the specified device."""
    model = MFDNet(m=4, k=3, c=48).to(device)
    return model


def infer(model, device):
    """Perform inference and measure the time taken."""
    # Create a fixed input tensor for consistent benchmarking
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # Perform inference
    N = 100
    with torch.no_grad():
        start_time = time.time()  # Start timer
        for _ in range(N):
            output_tensor = model(input_tensor)
        end_time = time.time()  # End timer

    inference_time = end_time - start_time
    return inference_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    # Benchmark without switch_to_deploy()
    model.eval()  # Set to evaluation mode
    time_without_deploy = infer(model, device)
    print(f"Inference time without switch_to_deploy(): {time_without_deploy:.4f} seconds")

    # Benchmark with switch_to_deploy()
    for module in model.modules():
        if isinstance(module, RepConv):
            module.switch_to_deploy()
    time_with_deploy = infer(model, device)
    print(f"Inference time with switch_to_deploy(): {time_with_deploy:.4f} seconds")

    # Compare the results
    improvement = (time_without_deploy - time_with_deploy) / time_without_deploy * 100
    print(f"Improvement in inference time: {improvement:.2f}%")


if __name__ == "__main__":
    main()


"

Inference time without switch_to_deploy(): 6.0312 seconds
Inference time with switch_to_deploy(): 4.6921 seconds
Improvement in inference time: 22.20%

"
