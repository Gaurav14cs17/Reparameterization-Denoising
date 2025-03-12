# **Implementation of the Mobile Real-Time Denoising Network (MFDNet)**  

Original Paper: [Lightweight network towards real-time image denoising on mobile devices](https://arxiv.org/abs/2211.04687)  

---

## **1. Training**  

1. Modify the configuration parameters in `train.py` according to your dataset's storage path.  
2. Run `train.py` to start training.  

### **Dataset Used**  
- This implementation uses **"Train400"** as the training dataset.  
- No separate validation or test set is defined.  
- If needed, you can modify the image loading code in `./utils/utils.py` and adjust `train.py` accordingly.  

---

## **2. Testing**  

1. Modify the relevant parameters in `train.py`.  
2. Run `test.py` to perform testing.  

> **Note:** Training and testing configurations are not written separately.  

---

## **3. Notes on Implementation**  
To explain the provided code mathematically, we need to break down the key components and operations involved in the **MFDNet** architecture, including the **Haar wavelet transform**, **RepConv layers**, **Multi-Frequency Attention (MFA)**, and **weight fusion**. Let's go through each component step by step.

---

### **1. Haar Wavelet Transform**

The **Haar wavelet transform** is used for downsampling the input image. It decomposes the image into four sub-bands:
- **LL (Low-Low)**: Approximation coefficients (low-frequency components).
- **LH (Low-High)**: Horizontal detail coefficients.
- **HL (High-Low)**: Vertical detail coefficients.
- **HH (High-High)**: Diagonal detail coefficients.

Mathematically, the Haar transform for a 2D image \( I \) can be represented as:

\[
\begin{aligned}
\text{LL} &= \frac{I_{2i,2j} + I_{2i,2j+1} + I_{2i+1,2j} + I_{2i+1,2j+1}}{4}, \\
\text{LH} &= \frac{I_{2i,2j} + I_{2i,2j+1} - I_{2i+1,2j} - I_{2i+1,2j+1}}{4}, \\
\text{HL} &= \frac{I_{2i,2j} - I_{2i,2j+1} + I_{2i+1,2j} - I_{2i+1,2j+1}}{4}, \\
\text{HH} &= \frac{I_{2i,2j} - I_{2i,2j+1} - I_{2i+1,2j} + I_{2i+1,2j+1}}{4}.
\end{aligned}
\]

In the code, the Haar transform is applied twice to achieve **4x downsampling**.

---

### **2. RepConv (Reparameterizable Convolution)**

The **RepConv** block consists of three convolutional layers:
1. A **1x1 convolution** to expand the number of channels.
2. A **3x3 convolution** for spatial feature extraction.
3. A **1x1 convolution** to reduce the number of channels back to the original size.

Mathematically, the RepConv operation can be expressed as:

\[
\text{RepConv}(x) = W_3 \ast (W_2 \ast (W_1 \ast x + b_1) + b_2) + b_3,
\]

where:
- \( W_1, W_2, W_3 \) are the weights of the 1x1, 3x3, and 1x1 convolutions, respectively.
- \( b_1, b_2, b_3 \) are the biases of the corresponding convolutions.
- \( \ast \) denotes the convolution operation.

The output of RepConv is added to the input for **residual learning**:

\[
\text{Output} = \text{RepConv}(x) + x.
\]

---

### **3. Multi-Frequency Attention (MFA)**

The **MFA** module applies a learnable attention map to the input features. It consists of:
1. A **Haar wavelet transform** to decompose the input into multi-frequency components.
2. A series of **convolutional layers** to compute the attention map.
3. An **interpolation** step to upsample the attention map.

Mathematically, the MFA operation can be expressed as:

\[
\text{MFA}(x) = \text{Interpolate}(\text{Conv}(\text{Haar}(x))) \odot x,
\]

where:
- \( \text{Haar}(x) \) is the Haar wavelet transform of \( x \).
- \( \text{Conv} \) represents the convolutional layers.
- \( \text{Interpolate} \) upsamples the attention map to match the spatial dimensions of \( x \).
- \( \odot \) denotes element-wise multiplication.

---

### **4. Weight Fusion**

The **weight fusion** process combines the weights of the 1x1 and 3x3 convolutions in RepConv into a single 3x3 convolution. This is done to simplify the model for deployment.

#### **Fusion of 1x1 and 3x3 Convolutions**

Let \( W_1 \) and \( W_2 \) be the weights of the 1x1 and 3x3 convolutions, respectively. The fused weight \( W_{12} \) is computed as:

\[
W_{12} = W_2 \ast W_1,
\]

where \( \ast \) denotes the convolution operation. The fused bias \( b_{12} \) is computed as:

\[
b_{12} = W_2 \ast b_1 + b_2.
\]

#### **Fusion with Final 1x1 Convolution**

Let \( W_3 \) be the weight of the final 1x1 convolution. The fused weight \( W_{123} \) is computed as:

\[
W_{123} = W_3 \ast W_{12},
\]

and the fused bias \( b_{123} \) is computed as:

\[
b_{123} = W_3 \ast b_{12} + b_3.
\]

#### **Identity Mapping**

To account for the residual connection, an identity mapping is added to the fused weight:

\[
W_{\text{fused}} = W_{123} + I,
\]

where \( I \) is the identity matrix padded to match the dimensions of \( W_{123} \).

---

### **5. Overall Forward Pass**

The forward pass of the **MFDNet** can be summarized as:

1. **Downsampling**:
   \[
   x_{\text{down}} = \text{Haar}(\text{Haar}(x)).
   \]

2. **MFDB Blocks**:
   \[
   x_{\text{features}} = x_{\text{down}} + \sum_{i=1}^m \text{MFDB}_i(x_{\text{down}}).
   \]

3. **Reconstruction**:
   \[
   x_{\text{recon}} = \text{PixelShuffle}(\text{Conv}(x_{\text{features}})).
   \]

4. **Residual Learning**:
   \[
   \text{Output} = x_{\text{recon}} + x.
   \]

---

### **6. PlainDN (Plain Denoising Network)**

The **PlainDN** replaces the RepConv layers with standard convolutions. The forward pass is similar to MFDNet, but the MFDB blocks use plain convolutions instead of RepConv.

---

### **Conclusion**

The **MFDNet** architecture leverages:
1. **Haar wavelet transforms** for efficient downsampling.
2. **RepConv layers** for lightweight feature extraction.
3. **Multi-Frequency Attention (MFA)** for adaptive feature refinement.
4. **Weight fusion** to simplify the model for deployment.

Mathematically, the operations are designed to balance **computational efficiency** and **denoising performance**, making the network suitable for real-time applications on mobile devices.

### **(1) MFA Module Downsampling Method**  
- The paper does not specify whether the MFA module uses **Haar wavelet transform** or **convolution** for downsampling.  
- It also does not explain how the number of channels changes (only mentioning that the MFA width is **1/4 of the model**).  
- In this implementation, **Haar wavelet transform** is used for downsampling.  

### **(2) Input Image Channels**  
- Based on the additional channels introduced by the Haar transform and the model structure in the paper, the original implementation likely uses **three-channel RGB images** as input.  
- However, since inference on an **iPhone 11 NPU takes 14.2ms**, and considering personal needs, this implementation uses **single-channel grayscale images** as input.  
- The model's channel structure was adjusted accordingly.  

### **(3) Reparameterization**  
- The paper briefly mentions parameter reinitialization but does not provide detailed explanations.  
- The report I referenced mentioned **ECB . Reparameterization**, which differs from the original paper's method by only **0.02dB**.  
- Thus, **ECB. Reparameterization** was adopted in this implementation.
-  ![Parameter Reinitialization](https://github.com/Gaurav14cs17/Reparameterization-Denoising/blob/main/images/image_1.png )
  ### Paper:
  - [Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution](https://arxiv.org/pdf/2204.08397)
  - [PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution](https://arxiv.org/html/2409.13435v1)
  - [Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf)
  - [awesome-reparameterize](https://github.com/pdh930105/awesome-reparameterize)
  - [RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization](https://github.com/suous/RepNeXt)

----
> ### Model Reference:
  > Haar transform code was adapted from [MWCNNv2](https://github.com/lpj-github-io/MWCNNv2/blob/master/MWCNN_code/model/common.py).  

---

## **4. Conclusion**  
- **Performance:** The model performs well in terms of both **speed and denoising quality**.
- ![Parameter Reinitialization](https://github.com/Gaurav14cs17/Reparameterization-Denoising/blob/main/images/image_2.png )

---

### **Summary**  
It delivers **efficient real-time denoising** while maintaining high performance.  

