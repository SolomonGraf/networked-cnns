import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from preprocessor import Preprocessor
from image_generator import JellyBeanGenerator
import os, shutil
from trainer import Trainer

class EllipseCounter(nn.Module):
    def __init__(self):
        super(EllipseCounter, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 4 pooling layers of stride 2, the 512x512 image becomes 32x32
        # (512 / 2^4 = 32)
        # Final conv output is 256 channels of 32x32
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 256x256
        x = self.pool(F.relu(self.conv2(x)))  # 128x128
        x = self.pool(F.relu(self.conv3(x)))  # 64x64
        x = self.pool(F.relu(self.conv4(x)))  # 32x32
        
        x = x.view(-1, 256 * 32 * 32)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class Agent:
    def __init__(self, path, id, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.path = path
        self.model = EllipseCounter().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.id = id
        if os.path.isfile(self.path):
            self.model.load_state_dict(torch.load(self.path))
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.path)

    def eval(self, img_path):
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch and channel dims (1, 1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)  # Convert to tensor
        image = image.to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(image).item()
        return prediction
    
    def random_train(self):
        dir = "random_train"
        shutil.rmtree(dir, ignore_errors=True)
        os.mkdir(dir)

        generator = JellyBeanGenerator(
            image_size=(512, 512),
            min_jellybeans=100,
            max_jellybeans=500,
            output_dir=dir
        )
        generator.generate_dataset(100)

        batch_size = 10
        train_loader, test_loader = Preprocessor.get_dataloaders(dir, "annotations.txt", batch_size)
        trainer = Trainer(self, train_loader, test_loader, epochs=1)
        trainer.train()

        shutil.rmtree(dir, ignore_errors=True)

    def train(self, img_path, count):
        self.model.train()
        return
    
    def copy(self, id, path):
        a = Agent(path = self.path, device=self.device, id=id)
        a.path = path
        return a