from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torch
import json
from PIL import Image

class EllipseDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.png')]
        with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
            self.annotations = json.load(f)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = self.transform(image)
        count = self.annotations[self.image_files[idx]]
        return image, torch.tensor(count, dtype=torch.float32)