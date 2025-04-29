from preprocessor import Preprocessor
from agent import Agent
from network import Network
from tqdm import tqdm
import sys

def network(path, folder):
    base = Agent(path, 0)
    base.load()

    network = Network.gen(n=40, d=4, base=base, folder=folder)

    for a in tqdm(network.agents.values()):
        a.random_train()
    
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

def eval(folder, res_json):
    n = Network(4, folder)
    n.eval("test.png", res_json)

def rt(path, img):
    base = Agent(path, 0)
    base.load()
    copy = base.deepcopy(1, "copy.pth")
    copy.load()
    
    cnt = base.eval(img)
    copy.reinforce(img, cnt)
    cntprime = copy.eval(img)

    print(f"Original Guess: {cnt} \nNew Guess: {cntprime}")

def reinforce_all(folder, img, res):
    n = Network(4, folder)
    res_path = "res.json"
    n.eval(img, res_path)
    n.reinforce_all(res_path, img)
    n.eval(img, res)


if __name__ == "__main__":
    if sys.argv[1] == "network":
        network(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "base":
        base(sys.argv[2])
    if sys.argv[1] == "eval":
        eval(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "rt":
        rt(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "reinforce":
        reinforce_all(sys.argv[2], sys.argv[3], sys.argv[4])