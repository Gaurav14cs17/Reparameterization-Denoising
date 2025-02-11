# **Unofficial Implementation of the Mobile Real-Time Denoising Network (MFDNet)**  

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

This project was completed some time ago, and I am now organizing and sharing it.  
At the time of implementation, the original paper had not yet been officially published (it was only available on arXiv).  
Some details in the paper were unclear, so certain parts of the implementation were based on educated guesses, including:  

### **(1) MFA Module Downsampling Method**  
- The paper does not specify whether the MFA module uses **Haar wavelet transform** or **convolution** for downsampling.  
- It also does not explain how the number of channels changes (only mentioning that the MFA width is **1/4 of the model**).  
- In this implementation, **Haar wavelet transform** is used for downsampling.  

### **(2) Input Image Channels**  
- Based on the additional channels introduced by the Haar transform and the model structure in the paper, the original implementation likely uses **three-channel RGB images** as input.  
- However, since inference on an **iPhone 11 NPU takes 14.2ms**, and considering personal needs, this implementation uses **single-channel grayscale images** as input.  
- The model's channel structure was adjusted accordingly.  

### **(3) Parameter Reinitialization**  
- The paper briefly mentions parameter reinitialization but does not provide detailed explanations.  
- The report I referenced mentioned **ECB parameter reinitialization**, which differs from the original paper's method by only **0.02dB**.  
- Thus, **ECB parameter reinitialization** was adopted in this implementation.  

> **Reference:** Haar transform code was adapted from [MWCNNv2](https://github.com/lpj-github-io/MWCNNv2/blob/master/MWCNN_code/model/common.py).  

---

## **4. Conclusion**  

- **Inference Speed:**  
  - **CPU:** **9ms** (480×360 image)  
  - **RTX 3060 GPU:** **1ms**  
  - **NCNN on mobile devices:** Achieves real-time inference  

- **Performance:** The model performs well in terms of both **speed and denoising quality**.  

---

### **Summary**  
This repository provides an **unofficial implementation** of **MFDNet** with some adjustments for mobile deployment.  
It delivers **efficient real-time denoising** while maintaining high performance.  

