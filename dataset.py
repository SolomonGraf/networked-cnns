from torch.utils.data import Dataset
import numpy as np
import os
import torch
from PIL import Image

class EllipseDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.annotations = {}
        
        # Read annotations
        with open(os.path.join(image_folder, annotation_file), 'r') as f:
            for line in f:
                filename, count = line.strip().split()
                self.annotations[filename] = int(count)
        
        self.image_files = list(self.annotations.keys())
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add channel dimension (1, H, W)
        
        # Get count
        count = self.annotations[img_name]
        count = torch.tensor(count, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, count