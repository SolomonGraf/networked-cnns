from preprocessor import Preprocessor
from agent import Agent
from tester import Tester
from network import Network
from tqdm import tqdm
import shutil, os, sys

def network(pth):
    # Configuration
    image_folder = "jellybeans_dataset"
    annotation_file = "annotations.json"
    batch_size = 32
    learning_rate = 0.001
    shutil.rmtree("models", ignore_errors=True)
    os.mkdir("models")
    
    # Initialize components
    _, test_loader = Preprocessor.get_dataloaders(image_folder, annotation_file, batch_size)
    base = Agent(learning_rate=learning_rate, id=0)

    

    # test has 345
    
    # Test
    tester = Tester(base, test_loader)
    tester.test()

    network = Network(n=10, d=4, base=base)

    for a in tqdm(network.agents):
        a.random_train()
        count = a.eval('test.png')
        print(count)
    
def base(path):
    # Configuration
    image_folder = "jellybeans_dataset"
    batch_size = 32

     # Initialize components
    train_loader, test_loader = Preprocessor.get_dataloaders(image_folder, batch_size)
    base = Agent(path)

    base.train(train_loader, test_loader)

    # Test
    tester = Tester(base, test_loader)
    tester.test()

    base.save()

def ltt(path):
    base = Agent(path)
    base.load()

    print("CNN estimates test.png at {0}. Real value is 345".format(base.eval("test.png")))


if __name__ == "__main__":
    if sys.argv[1] == "network":
        network("best_model.pth")
    if sys.argv[1] == "base":
        base("base.pth")
    if sys.argv[1] == "ltt":
        ltt("base.pth")