# Mobile Real-Time Denoising Network (MFDNet)

This repository contains the implementation of **MFDNet**, a lightweight and efficient denoising network designed for real-time performance on mobile devices. The network leverages **Haar wavelet transforms**, **RepConv layers**, and **Multi-Frequency Attention (MFA)** for high-quality image denoising.

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

\[
\text{Output} = \text{RepConv}(x) + x.
\]

### Multi-Frequency Attention (MFA)

The **MFA** module applies a learnable attention map to the input features. It consists of:
1. A **Haar wavelet transform** to decompose the input into multi-frequency components.
2. A series of **convolutional layers** to compute the attention map.
3. An **interpolation** step to upsample the attention map.

Mathematically, the MFA operation can be expressed as:

\[
\text{MFA}(x) = \text{Interpolate}(\text{Conv}(\text{Haar}(x))) \odot x,
\]

where \( \odot \) denotes element-wise multiplication.

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

---

## Mathematical Explanation

### Haar Wavelet Transform

The Haar transform for a 2D image \( I \) can be represented as:

\[
\begin{aligned}
\text{LL} &= \frac{I_{2i,2j} + I_{2i,2j+1} + I_{2i+1,2j} + I_{2i+1,2j+1}}{4}, \\
\text{LH} &= \frac{I_{2i,2j} + I_{2i,2j+1} - I_{2i+1,2j} - I_{2i+1,2j+1}}{4}, \\
\text{HL} &= \frac{I_{2i,2j} - I_{2i,2j+1} + I_{2i+1,2j} - I_{2i+1,2j+1}}{4}, \\
\text{HH} &= \frac{I_{2i,2j} - I_{2i,2j+1} - I_{2i+1,2j} + I_{2i+1,2j+1}}{4}.
\end{aligned}
\]

### RepConv Layers

The RepConv operation can be expressed as:

\[
\text{RepConv}(x) = W_3 \ast (W_2 \ast (W_1 \ast x + b_1) + b_2) + b_3,
\]

where \( W_1, W_2, W_3 \) are the weights of the 1x1, 3x3, and 1x1 convolutions, respectively.

### Weight Fusion

The fused weight \( W_{12} \) is computed as:

\[
W_{12} = W_2 \ast W_1,
\]

and the fused bias \( b_{12} \) is computed as:

\[
b_{12} = W_2 \ast b_1 + b_2.
\]

The final fused weight \( W_{\text{fused}} \) is:

\[
W_{\text{fused}} = W_{123} + I,
\]

where \( I \) is the identity matrix.

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
