import torch
import torch.nn as nn
import onnx
import numpy as np
from model.MFDNet import MFDNet
from model.ECBSR_MFDNet import ECBSR_MFDNET, ECBSR_MFDNET_PLAIN
from train import opt


def main():
    # Set device (GPU or CPU)
    use_cuda = opt.cuda >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{opt.cuda}' if use_cuda else 'cpu')

    if use_cuda:
        print(f"Using CUDA (GPU {opt.cuda}) for acceleration.")
        torch.cuda.set_device(opt.cuda)
        torch.backends.cudnn.benchmark = True  # Optimizes for fixed-size inputs
    else:
        print("Using CPU for training/testing.")

    # Convert only MFDNet model or both ECBSR + MFDNet
    only_mfdnet = True

    if only_mfdnet:
        try:
            # Load MFDNet model
            model = MFDNet(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)
            model.load_state_dict(torch.load(opt.test_model, map_location=device), strict=False)
            model.eval()

            # Define input tensor
            input_tensor = torch.randn(1, 1, 90, 120, device=device)

            # Export to ONNX
            torch.onnx.export(
                model, input_tensor, "./mfdnet-sigma8.onnx",
                input_names=["inputs"], output_names=["outputs"],
                dynamic_axes={"inputs": {2: "inputs_height", 3: "inputs_width"},
                              "outputs": {2: "outputs_height", 3: "outputs_width"}},
                opset_version=11
            )
            print("MFDNet model successfully converted to ONNX.")

        except Exception as e:
            print(f"Error converting MFDNet to ONNX: {e}")

    else:
        try:
            # Load ECBSR + MFDNet model
            model_ecbsr_mfdnet = ECBSR_MFDNET_PLAIN(4, 16, 'prelu', 4, 1, 3, 3, 16).to(device)
            model_ecbsr_mfdnet.load_state_dict(torch.load("./weights/ecbsr_mfdnet_plain/ecbsr_mfdnet_plain.pt", 
                                                           map_location=device), strict=False)
            model_ecbsr_mfdnet.eval()

            # Define input tensor
            input_tensor = torch.randn(1, 1, 90, 120, device=device)

            # Export to ONNX
            torch.onnx.export(
                model_ecbsr_mfdnet, input_tensor, "./ecbsr_mfdnet_plain-sigma8.onnx",
                input_names=["inputs"], output_names=["outputs"],
                dynamic_axes={"inputs": {2: "inputs_height", 3: "inputs_width"},
                              "outputs": {2: "outputs_height", 3: "outputs_width"}},
                opset_version=11
            )
            print("ECBSR + MFDNet model successfully converted to ONNX.")

        except Exception as e:
            print(f"Error converting ECBSR + MFDNet to ONNX: {e}")


if __name__ == "__main__":
    main()
