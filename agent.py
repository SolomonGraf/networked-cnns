import torch, os, copy
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
from image_generator import JellyBeanGenerator
from PIL import Image
from cnn import EllipseCounterCNN
from preprocessor import Preprocessor

class Agent:
    def __init__(self, path, id):
        self.id = id
        self.path = path
        self.model = EllipseCounterCNN()
        self.criterion = nn.MSELoss()  # Mean Squared Error for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def save(self):
        """Save model and optimizer state to self.path."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, self.path)

    def load(self):
        """Load model and optimizer state from self.path."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No model found at {self.path}")
        
        state = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def deepcopy(self, i: int, new_path: str):
        # Create new agent instance with the new path
        new_agent = Agent(new_path, i)
        
        # Deep copy model state
        new_agent.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        
        # Deep copy optimizer state
        new_agent.optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        
        # Copy other attributes
        new_agent.device = self.device
        new_agent.criterion = copy.deepcopy(self.criterion)
        
        # Ensure model is on the right device
        new_agent.model.to(self.device)
        
        # Save the copied model to the new path
        new_agent.save()
        
        return new_agent

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
    
    def train(self, folder, epochs = 10):
        train_loader, test_loader = Preprocessor.get_dataloaders(folder)

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

    def random_train(self, epochs=5):
        temp_dir = "temp_random_data"
        gen = JellyBeanGenerator(output_dir=temp_dir)
        gen.generate_dataset(100)
        self.train(temp_dir, epochs)

    def reinforce(self, image, count):
        """Fine-tunes the model on a single annotated image."""
        # Convert image to tensor
        image = Image.open(image).convert('L')  # Grayscale
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = transform(image).unsqueeze(0).to(self.device)  # [1, 1, H, W]
        
        # Ensure proper tensor shapes
        count = torch.tensor(count, dtype=torch.float32, device=self.device)  # Shape []
        self.model.train()
        
        self.optimizer.zero_grad()
        output = self.model(image)  # Should output shape [] (scalar)
        
        # Explicitly reshape if needed
        if output.dim() > 0:
            output = output.squeeze()  # Remove all singleton dimensions
            
        loss = self.criterion(output, count)  # Both shapes should now match
        loss.backward()
        self.optimizer.step()

    def eval(self, image_path):
        # 1. Load and preprocess the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Same as in your dataset
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension [1, 1, 512, 512]
        image_tensor = image_tensor.to(self.device)
        
        # 2. Run inference
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        # 3. Return the count (squeeze to scalar)
        return int(prediction.item())