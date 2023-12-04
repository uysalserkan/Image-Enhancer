"""Image dataset defination."""
import os
import glob

import cv2
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Image dataset for batch operations."""

    def __init__(self, folder_path: str, output_path: str, model):
        self.folder_path = folder_path
        self.output_path = output_path
        self.model = model
        self.image_paths = self._load_images()

    def _load_images(self):
        """Load images."""
        img_paths = sorted(glob.glob(os.path.join(self.folder_path, '*')))

        return img_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        imgname, extension = os.path.splitext(os.path.basename(image_path))

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        print(f"{image_path} image loaded.")

        _, _, output = self.model.enhance(
            img,
            False,
            False,
            True
        )

        save_path = os.path.join(self.output_path, f'{imgname}{extension}')
        cv2.imwrite(save_path, output)
        print(f"{save_path} image saved.")

        return image_path

