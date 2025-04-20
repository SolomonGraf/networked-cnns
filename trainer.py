import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, agent, train_loader, test_loader, epochs=5, patience=5):
        self.agent = agent
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
    
    def train_epoch(self):
        self.agent.model.train()
        running_loss = 0.0
        
        for images, counts in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.agent.device)
            counts = counts.to(self.agent.device).unsqueeze(1)
            
            self.agent.optimizer.zero_grad()
            
            outputs = self.agent.model(images)
            loss = self.agent.criterion(outputs, counts)
            
            loss.backward()
            self.agent.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    def validate(self):
        self.agent.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, counts in tqdm(self.test_loader, desc="Validating"):
                images = images.to(self.agent.device)
                counts = counts.to(self.agent.device).unsqueeze(1)
                
                outputs = self.agent.model(images)
                loss = self.agent.criterion(outputs, counts)
                
                running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(self.test_loader.dataset)
        return epoch_loss
    
    def train(self):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
                self.agent.save_model()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break