import os
import cv2 as cv
import torch
import numpy as np
import torch.nn as nn
import time
from model.MFDNet import MFDNet
from model.Plaindn import PlainDN
from train import opt
from utils.Rep_params import rep_params


def main():
    # Set device based on availability
    use_cuda = opt.cuda >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{opt.cuda}' if use_cuda else 'cpu')

    if use_cuda:
        print(f"Using CUDA (GPU {opt.cuda}) for acceleration.")
        torch.cuda.set_device(opt.cuda)
        torch.backends.cudnn.benchmark = True  # Optimizes performance for fixed-size inputs
    else:
        print("Using CPU for training/testing.")

    # Initialize models
    model = MFDNet(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)
    model_plain = PlainDN(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)

    # Load model weights
    try:
        print(f"Loading model: {opt.test_model}")
        model.load_state_dict(torch.load(opt.test_model, map_location=device), strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Reparameterize model
    model_plain = rep_params(model, model_plain, opt, device)

    # Test inference
    x = torch.randn(1, 1, 24, 24, device=device)
    result_plain = model_plain(x)
    result_original = model(x)

    # Print difference between models
    print("Model difference:", torch.norm(result_plain - result_original).item())


if __name__ == "__main__":
    main()
    exit(0)
