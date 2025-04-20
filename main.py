from preprocessor import Preprocessor
from agent import Agent
from tester import Tester
from trainer import Trainer
from network import Network
from tqdm import tqdm
import shutil, os

def main():
    # Configuration
    image_folder = "jellybeans_dataset"
    annotation_file = "annotations.txt"
    batch_size = 32
    learning_rate = 0.001
    shutil.rmtree("models", ignore_errors=True)
    os.mkdir("models")
    
    # Initialize components
    _, test_loader = Preprocessor.get_dataloaders(image_folder, annotation_file, batch_size)
    base = Agent(learning_rate=learning_rate, id=0, path="best_model.pth")

    copy = base.copy(base.id, base.path)
    
    print("Estimation Before RT")
    print(copy.eval('jellybeans_dataset/test.png'))
    copy.random_train()
    print("Estimation After RT")
    print(copy.eval('jellybeans_dataset/test.png'))
    
    # Test
    tester = Tester(base, test_loader)
    tester.test()

    network = Network(n=10, d=4, base=base)

    for a in tqdm(network.agents):
        a.random_train()
        count = a.eval('jellybeans_dataset/test.png')
        print(count)
    

if __name__ == "__main__":
    main()