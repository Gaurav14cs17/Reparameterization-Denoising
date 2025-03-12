import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.MFDNet import *
import os
import time

# Custom Dataset Class
class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.clean_images = sorted(os.listdir(clean_dir))
        self.noisy_images = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])

        clean_image = Image.open(clean_path).convert("RGB")
        noisy_image = Image.open(noisy_path).convert("RGB")

        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)

        return noisy_image, clean_image


# Training Function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Training Loss: {epoch_loss:.4f}")
    return epoch_loss


# Evaluation Function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for noisy_images, clean_images in val_loader:
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)

            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


# Main Training Loop
def main():
    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = DenoisingDataset(
        clean_dir="path/to/clean_images/train",
        noisy_dir="path/to/noisy_images/train",
        transform=transform,
    )
    val_dataset = DenoisingDataset(
        clean_dir="path/to/clean_images/val",
        noisy_dir="path/to/noisy_images/val",
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = MFDNet(m=4, k=3, c=48).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "mfdnet_denoising.pth")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
