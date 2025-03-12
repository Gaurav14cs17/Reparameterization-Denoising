# Mobile Real-Time Denoising Network (MFDNet)

This repository contains the implementation of **MFDNet**, a lightweight and efficient denoising network designed for real-time performance on mobile devices. The network leverages **Haar wavelet transforms**, **RepConv layers**, and **Multi-Frequency Attention (MFA)** for high-quality image denoising.

Original Paper: [Lightweight network towards real-time image denoising on mobile devices](https://arxiv.org/abs/2211.04687)  

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
   - [Haar Wavelet Transform](#haar-wavelet-transform)
   - [RepConv Layers](#repconv-layers)
   - [Multi-Frequency Attention (MFA)](#multi-frequency-attention-mfa)
   - [Weight Fusion](#weight-fusion)
4. [Usage](#usage)
   - [Installation](#installation)
   - [Training](#training)
   - [Inference](#inference)
5. [Mathematical Explanation](#mathematical-explanation)
6. [Implementation Details](#implementation-details)
   - [Training](#training-details)
   - [Testing](#testing-details)
   - [Implementation Notes](#implementation-notes)
7. [License](#license)

---

## Overview

MFDNet is designed for **real-time denoising** on mobile devices. It uses a combination of **Haar wavelet transforms** for downsampling, **RepConv layers** for efficient feature extraction, and **Multi-Frequency Attention (MFA)** for adaptive feature refinement. The network is optimized for **low computational cost** while maintaining high denoising performance.

 ![Parameter Reinitialization](https://github.com/Gaurav14cs17/Reparameterization-Denoising/blob/main/images/image_1.png )

---

## Key Features

- **Haar Wavelet Transform**: Efficient downsampling and multi-frequency decomposition.
- **RepConv Layers**: Lightweight and reparameterizable convolutional layers.
- **Multi-Frequency Attention (MFA)**: Adaptive feature refinement using attention mechanisms.
- **Weight Fusion**: Simplifies the model for deployment by fusing convolution weights.
- **Real-Time Performance**: Optimized for mobile devices with low latency.

---

## Architecture

### Haar Wavelet Transform

The Haar wavelet transform is used for **4x downsampling**. It decomposes the input image into four sub-bands:
- **LL (Low-Low)**: Approximation coefficients (low-frequency components).
- **LH (Low-High)**: Horizontal detail coefficients.
- **HL (High-Low)**: Vertical detail coefficients.
- **HH (High-High)**: Diagonal detail coefficients.

### RepConv Layers

The **RepConv** block consists of three convolutional layers:
1. A **1x1 convolution** to expand the number of channels.
2. A **3x3 convolution** for spatial feature extraction.
3. A **1x1 convolution** to reduce the number of channels back to the original size.

The output of RepConv is added to the input for **residual learning**:



### Multi-Frequency Attention (MFA)

The **MFA** module applies a learnable attention map to the input features. It consists of:
1. A **Haar wavelet transform** to decompose the input into multi-frequency components.
2. A series of **convolutional layers** to compute the attention map.
3. An **interpolation** step to upsample the attention map.

Mathematically, the MFA operation can be expressed as:




### Weight Fusion

The **weight fusion** process combines the weights of the 1x1 and 3x3 convolutions in RepConv into a single 3x3 convolution. This simplifies the model for deployment.

---

## Usage

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MFDNet.git
   cd MFDNet
   ```

2. Install the required dependencies:
   ```bash
   pip install torch torchvision torchaudio matplotlib opencv-python
   ```

### Training

To train the MFDNet model, use the following command:
```bash
python train.py --m 4 --k 3 --c 64
```
- `--m`: Number of MFDB blocks.
- `--k`: Number of RepConv layers per MFDB.
- `--c`: Number of channels.

### Inference

To perform inference on an image, use the following command:
```bash
python inference.py --input input_image.png --output output_image.png
```



```

import torch
import torch.nn as nn
import torch.nn.functional as F

class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(RepConv, self).__init__()
        self.training_mode = True  # Track whether in training or inference mode

        # Training-time components (expand-and-squeeze topology)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Identity mapping for residual connection
        self.identity = nn.Identity() if in_channels == out_channels and stride == 1 else None

        # Placeholder for inference-time fused convolution
        self.rep_conv = None

    def forward(self, x):
        if self.training_mode:
            # Training-time forward pass
            out = self.conv1x1_1(x) + self.conv3x3(x) + self.conv1x1_2(x)
            if self.identity is not None:
                out += self.identity(x)
            return self.bn(out)
        else:
            # Inference-time forward pass
            return self.rep_conv(x)

    def fuse(self):
        """Merge the multi-branch convolutions into a single 3x3 kernel for inference."""
        with torch.no_grad():
            # Fuse the 1x1 and 3x3 kernels
            kernel3x3 = self.conv3x3.weight.clone()
            kernel1x1_1 = F.pad(self.conv1x1_1.weight, [1, 1, 1, 1])  # Pad 1x1 to 3x3
            kernel1x1_2 = F.pad(self.conv1x1_2.weight, [1, 1, 1, 1])  # Pad 1x1 to 3x3
            fused_kernel = kernel3x3 + kernel1x1_1 + kernel1x1_2

            # Fuse the batch normalization into the convolution
            bn_mean = self.bn.running_mean
            bn_var = self.bn.running_var
            bn_weight = self.bn.weight
            bn_bias = self.bn.bias
            bn_eps = self.bn.eps

            # Compute the fused convolution weights and bias
            fused_weight = fused_kernel * (bn_weight / torch.sqrt(bn_var + bn_eps)).view(-1, 1, 1, 1)
            fused_bias = bn_bias - bn_mean * bn_weight / torch.sqrt(bn_var + bn_eps)

            # Create the final inference-time convolution
            self.rep_conv = nn.Conv2d(
                self.conv3x3.in_channels,
                self.conv3x3.out_channels,
                kernel_size=3,
                stride=self.conv3x3.stride,
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
        del self.conv1x1_1, self.conv3x3, self.conv1x1_2, self.bn,    self.identity
        self.conv1x1_1, self.conv3x3, self.conv1x1_2, self.bn, self.identity = None, None, None, None, None



if __name__ == "__main__":
    model = RepConv(in_channels=64, out_channels=64)
    x = torch.randn(1, 64, 32, 32)
    print("Training output shape:", model(x).shape)

    model.switch_to_deploy()
    print("Inference output shape:", model(x).shape)
  
```

---



## Mathematical Explanation






### RepConv Layers



---

## Implementation Details

### Training

1. Modify the configuration parameters in `train.py` according to your dataset's storage path.
2. Run `train.py` to start training.

#### Dataset Used
- This implementation uses **"Train400"** as the training dataset.
- No separate validation or test set is defined.
- If needed, you can modify the image loading code in `./utils/utils.py` and adjust `train.py` accordingly.

### Testing

1. Modify the relevant parameters in `train.py`.
2. Run `test.py` to perform testing.

---

## Implementation Notes

### (1) MFA Module Downsampling Method
- The paper does not specify whether the MFA module uses **Haar wavelet transform** or **convolution** for downsampling.
- It also does not explain how the number of channels changes (only mentioning that the MFA width is **1/4 of the model**).
- In this implementation, **Haar wavelet transform** is used for downsampling.

### (2) Input Image Channels
- Based on the additional channels introduced by the Haar transform and the model structure in the paper, the original implementation likely uses **three-channel RGB images** as input.
- However, since inference on an **iPhone 11 NPU takes 14.2ms**, and considering personal needs, this implementation uses **single-channel grayscale images** as input.
- The model's channel structure was adjusted accordingly.

### (3) Reparameterization
- The paper briefly mentions parameter reinitialization but does not provide detailed explanations.
- The report I referenced mentioned **ECB . Reparameterization**, which differs from the original paper's method by only **0.02dB**.
- Thus, **ECB. Reparameterization** was adopted in this implementation.

![Parameter Reinitialization](https://github.com/Gaurav14cs17/Reparameterization-Denoising/blob/main/images/image_2.png )

#### References for Reparameterization:
- [Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution](https://arxiv.org/pdf/2204.08397)
- [PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution](https://arxiv.org/html/2409.13435v1)
- [Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf)
- [awesome-reparameterize](https://github.com/pdh930105/awesome-reparameterize)
- [RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization](https://github.com/suous/RepNeXt)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by lightweight denoising architectures and reparameterization techniques.
- Thanks to the open-source community for contributions to deep learning and image processing.

---

For questions or feedback, please open an issue or contact the maintainers.

---

### **Conclusion**

The **MFDNet** architecture leverages:
1. **Haar wavelet transforms** for efficient downsampling.
2. **RepConv layers** for lightweight feature extraction.
3. **Multi-Frequency Attention (MFA)** for adaptive feature refinement.
4. **Weight fusion** to simplify the model for deployment.

Mathematically, the operations are designed to balance **computational efficiency** and **denoising performance**, making the network suitable for real-time applications on mobile devices.
