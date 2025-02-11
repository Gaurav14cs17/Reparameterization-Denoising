import torch
import torch.nn as nn
import torch.nn.functional as F

def rep_params(model, model_plain, opt, device):
    """Replace parameters from `model` into `model_plain` using weight fusion.

    Args:
        model (nn.Module): Original model with RepConv layers.
        model_plain (nn.Module): Plain model (without RepConv).
        opt (object): Contains parameters M_MFDB (number of MFDB blocks) 
                      and K_RepConv (number of RepConv layers per block).
        device (torch.device): Device where computations are performed.

    Returns:
        nn.Module: Updated model_plain with fused weights.
    """
    state_dict_model = model.state_dict()
    state_dict_model_plain = model_plain.state_dict()

    for key, value in state_dict_model_plain.items():
        if key in state_dict_model:
            # Directly copy matching parameters
            state_dict_model_plain[key] = state_dict_model[key]
        else:
            # Process RepConv layers using weight fusion
            for m in range(opt.M_MFDB):
                for k in range(opt.K_RepConv):
                    key_weight_plain = f'M_block.{m}.mfdb.{k}.weight'
                    key_bias_plain = f'M_block.{m}.mfdb.{k}.bias'

                    if key == key_weight_plain:
                        # Extract weights and biases from repconv sequential layers
                        Ka = state_dict_model[f'M_block.{m}.mfdb.{k}.repconv.0.weight']
                        Ba = state_dict_model[f'M_block.{m}.mfdb.{k}.repconv.0.bias']
                        Kb = state_dict_model[f'M_block.{m}.mfdb.{k}.repconv.1.weight']
                        Bb = state_dict_model[f'M_block.{m}.mfdb.{k}.repconv.1.bias']
                        Kc = state_dict_model[f'M_block.{m}.mfdb.{k}.repconv.2.weight']
                        Bc = state_dict_model[f'M_block.{m}.mfdb.{k}.repconv.2.bias']

                        # Fuse 1x1 and 3x3 convolution (Ka * Kb)
                        Kab = F.conv2d(Kb, Ka.permute(1, 0, 2, 3))
                        Bab = torch.ones(1, Ka.shape[0], 3, 3, device=device) * Ba.view(1, -1, 1, 1)
                        Bab = F.conv2d(Bab, Kb).view(-1,) + Bb

                        # Fuse previous result with final 1x1 convolution (Kabc)
                        out_channels, in_channels, _, _ = Kc.shape
                        Kabc = torch.zeros_like(Kc, device=device)
                        Babc = torch.zeros(out_channels, device=device)

                        for i in range(out_channels):
                            Kabc[i] = torch.sum(Kab * Kc[i].unsqueeze(1), dim=0)
                            Babc[i] = Bc[i] + torch.sum(Bab * Kc[i].squeeze(1).squeeze(1))

                        # Add identity mapping with padding for residual learning
                        identity = torch.eye(in_channels).view(in_channels, in_channels, 1, 1).to(device)
                        identity = F.pad(identity, pad=(1, 1, 1, 1), mode='constant', value=0)

                        # Store updated weights in model_plain
                        state_dict_model_plain[key_weight_plain] = Kabc + identity
                        state_dict_model_plain[key_bias_plain] = Babc

    # Load updated parameters into model_plain
    model_plain.load_state_dict(state_dict_model_plain)
    return model_plain
