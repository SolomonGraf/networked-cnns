from preprocessor import Preprocessor
from agent import Agent
from trainer import Trainer
from tester import Tester
from network import Network
from tqdm import tqdm

def main():
    # Configuration
    image_folder = "jellybeans_dataset"
    annotation_file = "annotations.txt"
    batch_size = 32
    learning_rate = 0.001
    
    # Initialize components
    _, test_loader = Preprocessor.get_dataloaders(image_folder, annotation_file, batch_size)
    base = Agent(learning_rate=learning_rate, id=0, path="best_model.pth")
    
    # Test
    tester = Tester(base, test_loader)
    tester.test()

    network = Network(n=40, d=4, base=base)

    for a in tqdm(network.agents):
        a.random_train()
        count = a.eval('jellybeans_dataset/test.png')
        print(count)
    

if __name__ == "__main__":
    main()