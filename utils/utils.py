import torch
import torch.utils.data as data
import os
import cv2 as cv
import numpy as np
import datetime


class JLDatasetFromMat(data.Dataset):
    def __init__(self, file_path, sigma, img_size):
        super(JLDatasetFromMat, self).__init__()
        self.sigma = sigma
        self.file_path = file_path
        self.img_size = img_size
        self.img_names = os.listdir(file_path)
        self.rnd_aug = np.random.randint(8, size=len(self.img_names))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.file_path, self.img_names[index])
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        img = cv.resize(img, self.img_size, interpolation=cv.INTER_CUBIC)
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)[:, :, 0]  # Extract Y channel

        img_y_aug = data_augmentation(img, self.rnd_aug[index])
        img_y_aug = np.expand_dims(img_y_aug, axis=0) / 255.0  # Normalize to [0,1]

        noise = np.random.normal(0, self.sigma / 255.0, size=img_y_aug.shape)
        noisy_img = img_y_aug + noise

        return torch.from_numpy(noisy_img).float(), torch.from_numpy(img_y_aug).float()


# Optimized Data Augmentation
def data_augmentation(image, mode=0):
    """Applies one of 8 augmentation modes to an image."""
    augmentations = {
        0: lambda x: x,                       # Original
        1: np.flipud,                          # Flip up-down
        2: lambda x: np.rot90(x, k=1),         # Rotate 90°
        3: lambda x: np.flipud(np.rot90(x)),   # Rotate 90° + Flip
        4: lambda x: np.rot90(x, k=2),         # Rotate 180°
        5: lambda x: np.flipud(np.rot90(x, k=2)), # Rotate 180° + Flip
        6: lambda x: np.rot90(x, k=3),         # Rotate 270°
        7: lambda x: np.flipud(np.rot90(x, k=3))  # Rotate 270° + Flip
    }
    return augmentations.get(mode, lambda x: x)(image)


# Optimized Timestamp Function
def cur_timestamp_str():
    """Returns the current timestamp as a formatted string."""
    return datetime.datetime.now().strftime("%Y-%m%d-%H%M")
