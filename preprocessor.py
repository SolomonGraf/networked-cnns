from torch.utils.data import DataLoader, random_split
from dataset import EllipseDataset

class Preprocessor:
    @staticmethod
    def get_dataloaders(image_folder, batch_size=32, train_ratio=0.8):
        full_dataset = EllipseDataset(image_folder)
        
        # Split dataset
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader