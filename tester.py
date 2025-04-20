import torch
from tqdm import tqdm
import numpy as np

class Tester:
    def __init__(self, agent, test_loader):
        self.agent = agent
        self.test_loader = test_loader
    
    def test(self):
        self.agent.model.eval()
        test_loss = 0.0
        predictions = []
        actual_counts = []
        
        with torch.no_grad():
            for images, counts in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.agent.device)
                counts = counts.to(self.agent.device).unsqueeze(1)
                
                outputs = self.agent.model(images)
                loss = self.agent.criterion(outputs, counts)
                
                test_loss += loss.item() * images.size(0)
                predictions.extend(outputs.cpu().numpy().flatten())
                actual_counts.extend(counts.cpu().numpy().flatten())
        
        test_loss = test_loss / len(self.test_loader.dataset)
        
        # Calculate MAE
        mae = np.mean(np.abs(np.array(predictions) - np.array(actual_counts)))
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return test_loss, mae, predictions, actual_counts