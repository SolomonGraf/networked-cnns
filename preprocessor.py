from torch.utils.data import DataLoader, random_split
from dataset import EllipseDataset
import torch

class Preprocessor:
    @staticmethod
    def get_dataloaders(folder, batch_size=32, train_ratio=0.8, seed=None):
        dataset = EllipseDataset(folder)

        if seed is not None:
            torch.manual_seed(seed)
        
        # Split dataset
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader