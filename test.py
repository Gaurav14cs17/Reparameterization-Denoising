import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# Custom Dataset Class for Testing
class DenoisingTestDataset(Dataset):
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

        return noisy_image, clean_image, self.clean_images[idx]


# Load the Trained Model
def load_model(model_path, device):
    model = MFDNet(m=4, k=3, c=48).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model


# Perform Inference and Calculate Metrics
def test(model, test_loader, device, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)  # Create directory to save results
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for noisy_images, clean_images, image_names in test_loader:
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Perform inference
            denoised_images = model(noisy_images)

            # Convert tensors to numpy arrays for metric calculation
            denoised_images = denoised_images.cpu().numpy()
            clean_images = clean_images.cpu().numpy()

            # Calculate PSNR and SSIM for each image in the batch
            for i in range(denoised_images.shape[0]):
                denoised = denoised_images[i].transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
                clean = clean_images[i].transpose(1, 2, 0)

                # Clip values to [0, 1] for valid PSNR and SSIM calculation
                denoised = np.clip(denoised, 0, 1)
                clean = np.clip(clean, 0, 1)

                # Calculate PSNR and SSIM
                psnr_value = psnr(clean, denoised, data_range=1)
                ssim_value = ssim(clean, denoised, data_range=1, multichannel=True)

                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)

                # Save the denoised image
                denoised_image = (denoised * 255).astype(np.uint8)
                denoised_image = Image.fromarray(denoised_image)
                denoised_image.save(os.path.join(save_dir, image_names[i]))

    # Calculate average PSNR and SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")


# Main Testing Script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "mfdnet_denoising.pth"  # Path to the trained model

    # Load the model
    model = load_model(model_path, device)

    # Dataset and DataLoader for testing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = DenoisingTestDataset(
        clean_dir="path/to/clean_images/test",
        noisy_dir="path/to/noisy_images/test",
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Perform testing
    test(model, test_loader, device, save_dir="results")


if __name__ == "__main__":
    main()
