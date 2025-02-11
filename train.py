import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import jl_DatasetFromMat, cur_timestamp_str
from model.MFDNet import MFDNet

parser = argparse.ArgumentParser(description="PyTorch MFDNet Training")

# Training Parameters
parser.add_argument("--batchSize", type=int, default=40, help="Training batch size")
parser.add_argument("--epoch", type=int, default=100000, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0004, help="Learning Rate")
parser.add_argument("--dataset", type=str, default="/home/jl/Project/DnCNN-unofficial/Train400", help="Path to training dataset")
parser.add_argument("--sigma", type=int, default=8, help="Gaussian noise level")
parser.add_argument("--pretrain", type=str, default=None, help="Path of pretrained model")
parser.add_argument("--checkpoint", type=str, default="./weights", help="Path to save model checkpoints")
parser.add_argument("--img_size", type=tuple, default=(360, 360), help="Training image size (Width, Height)")
parser.add_argument("--per_epoch_save", type=int, default=1000, help="Save model every N epochs")

# Testing Parameters
parser.add_argument("--test_input", type=str, default="./input_imgs", help="Test image input folder")
parser.add_argument("--test_output", type=str, default="./output_imgs/noise11", help="Test image output folder")
parser.add_argument("--test_model", type=str, default="./weights/mfdnet-m3k3c16-1209-1738-noise11/best_model.pt")

# MFDNet Parameters
parser.add_argument("--M_MFDB", type=int, default=3, help="Number of MFDB modules")
parser.add_argument("--K_RepConv", type=int, default=3, help="Number of RepConv layers per MFDB")
parser.add_argument("--c_channel", type=int, default=16, help="Number of channels in conv3x3 layers")

# Hardware Specification
parser.add_argument("--cuda", type=int, default=1, help="CUDA ID (-1 for CPU, 0, 1, ... for GPU)")
parser.add_argument("--threads", type=int, default=1, help="Number of data loading threads")

opt = parser.parse_args()


def set_device():
    """Set the computing device (GPU or CPU)."""
    use_cuda = opt.cuda >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{opt.cuda}" if use_cuda else "cpu")

    if use_cuda:
        print(f"Using CUDA (GPU {opt.cuda}) for acceleration.")
        torch.cuda.set_device(opt.cuda)
        torch.backends.cudnn.benchmark = True  # Optimizes performance for fixed-size input
    else:
        print("Using CPU for training/testing.")

    return device


def train(model, train_dataloader, optimizer, scheduler, loss_func, device, experiment_path):
    """Train the model with the specified parameters."""
    best_loss = float("inf")
    best_epoch = 0
    total_steps = len(train_dataloader.dataset) // opt.batchSize

    for epoch in range(opt.epoch):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for iter, (noise_img, clear_img) in enumerate(train_dataloader, 1):
            noise_img, clear_img = noise_img.to(device), clear_img.to(device)

            optimizer.zero_grad()
            out_img = model(noise_img)
            loss = loss_func(out_img, clear_img) / (opt.batchSize * 2)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss)

        avg_loss = epoch_loss / len(train_dataloader)
        duration = time.time() - start_time

        # Print training log
        print(f"--------------------------------")
        print(f"Epoch {epoch}: {total_steps} steps, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]}, Time: {duration:.2f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model_path = os.path.join(experiment_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)

        print(f"Best Epoch: {best_epoch}, Best Loss: {best_loss:.4f}")

        # Save model periodically
        if (epoch + 1) % opt.per_epoch_save == 0:
            checkpoint_path = os.path.join(experiment_path, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)


def main():
    device = set_device()
    torch.set_num_threads(opt.threads)

    # Load dataset and dataloader
    train_dataset = jl_DatasetFromMat(opt.dataset, opt.sigma, opt.img_size)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                                  num_workers=opt.threads, drop_last=True)

    # Initialize model, optimizer, scheduler, and loss function
    model = MFDNet(opt.M_MFDB, opt.K_RepConv, opt.c_channel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.epoch // 10, gamma=0.5)
    loss_func = nn.MSELoss(reduction="sum")

    # Load pretrained model if available
    if opt.pretrain:
        try:
            model.load_state_dict(torch.load(opt.pretrain, map_location=device))
            print(f"Loaded pretrained model: {opt.pretrain}")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
    else:
        print("Training the model from scratch.")

    # Create experiment folder
    timestamp = cur_timestamp_str()
    experiment_folder = f"mfdnet-m{opt.M_MFDB}k{opt.K_RepConv}c{opt.c_channel}-{timestamp}"
    experiment_path = os.path.join(opt.checkpoint, experiment_folder)
    os.makedirs(experiment_path, exist_ok=True)

    # Start training
    train(model, train_dataloader, optimizer, scheduler, loss_func, device, experiment_path)


if __name__ == "__main__":
    main()
