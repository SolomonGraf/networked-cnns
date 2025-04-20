from preprocessor import Preprocessor
from agent import Agent
from trainer import Trainer
from tester import Tester

def main():
    # Configuration
    image_folder = "jellybeans_dataset"
    annotation_file = "annotations.txt"
    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    
    # Initialize components
    train_loader, test_loader = Preprocessor.get_dataloaders(image_folder, annotation_file, batch_size)
    agent = Agent(learning_rate=learning_rate)
    
    # Train
    trainer = Trainer(agent, train_loader, test_loader, epochs=epochs)
    trainer.train()
    
    # Test
    tester = Tester(agent, test_loader)
    test_loss, mae, predictions, actual_counts = tester.test()
    
    # You can now use predictions and actual_counts for further analysis

if __name__ == "__main__":
    main()