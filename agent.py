import torch, os
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
from image_generator import JellyBeanGenerator
from PIL import Image

class EllipseCounterCNN(nn.Module):
    def __init__(self):
        super(EllipseCounterCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 512x512 → 512x512
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # 512x512 → 256x256
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 256x256 → 256x256
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # 256x256 → 128x128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 128x128 → 128x128
            nn.ReLU(),
            nn.MaxPool2d(2)                                       # 128x128 → 64x64
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
           )   # Regression output (ellipse count)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class Agent:
    def __init__(self, path):
        self.path = path
        self.model = EllipseCounterCNN()
        self.criterion = nn.MSELoss()  # Mean Squared Error for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        
        for images, counts in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            counts = counts.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, counts)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss
    
    def validate(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, counts in tqdm(test_loader, desc="Validating"):
                images = images.to(self.device)
                counts = counts.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, counts)
                
                running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(test_loader.dataset)
        return epoch_loss
    
    def train(self, train_loader, test_loader, epochs = 10):
        counter = 0
        patience = 2
        best_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(test_loader)
            
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                self.save()
            else:
                counter += 1
                if counter >= patience:
                    (f"Early stopping after {epoch + 1} epochs")
                    break

    def random_train(self, epochs=5, batch_size=4):
        temp_dir = "temp_random_data"
        gen = JellyBeanGenerator(output_dir=temp_dir)
        gen.generate_dataset(100)
        self.train(temp_dir, epochs, batch_size)

    def load(self):
        path = self.path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} does not exist.")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.to(self.device)
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def save(self):
        path = self.path
        """Save model and optimizer states to a .pth file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved successfully to {path}")

    def reinforce(self, image, count, epochs=1):
        """Fine-tunes the model on a single annotated image."""
        self.model.train()
        image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)  # Add batch dim
        count = torch.tensor([count], dtype=torch.float32).to(self.device)
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output.squeeze(), count)
            loss.backward()
            self.optimizer.step()

    def eval(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert("L")
        image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
        return round(output.item())  # Round to nearest integer (count)