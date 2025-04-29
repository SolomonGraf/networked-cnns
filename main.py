from preprocessor import Preprocessor
from agent import Agent
from network import Network
from tqdm import tqdm
import shutil, os, sys

def network(path):
    base = Agent(path, 0)
    base.load()

    network = Network(n=40, d=4, base=base)

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
    base = Agent(path, 0)

    base.train(train_loader, test_loader)
    print("CNN estimates test.png at {0}. Real value is 345".format(base.eval("test.png")))

    base.save()

def copytest(path, copy):
    base = Agent(path, 0)
    base.load()

    copy = base.deepcopy(1, copy)
    copy.save()
    copy.load()

    print("CNN estimates test.png at {0}. Real value is 345".format(copy.eval("test.png")))

def randomtest(path, copy):
    base = Agent(path, 0)
    base.load()

    copy = base.deepcopy(1, copy)
    copy.save()
    copy.load()
    
    copy.random_train()
    copy.save()

    print("CNN estimates test.png at {0}. Real value is 345".format(copy.eval("test.png")))


if __name__ == "__main__":
    if sys.argv[1] == "network":
        network(sys.argv[2])
    if sys.argv[1] == "base":
        base(sys.argv[2])