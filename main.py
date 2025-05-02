from agent import Agent
from network import Network
from control import Control
from tqdm import tqdm
from joblib import Parallel, delayed
from analysis import analyze as _analyze

import sys

def network(path, folder):  # n_jobs=-1 uses all available CPUs
    base = Agent(path, 0)
    base.load()

    network = Network.gen(n=40, d=4, base=base, folder=folder)

    def aux(a):
        network.agents[a].random_train()
        network.agents[a].save()

    with Parallel(n_jobs=-1) as p:
        p(
        delayed(aux)(a) 
        for a in tqdm(network.agents.keys())
        )

def control(path, folder):
    base = Agent(path, 0)
    base.load()

    network = Control.gen(n=40, base=base, folder=folder)

    for a in tqdm(network.agents.values(), desc="Randomizing Models"):
        a.random_train()
    
def base(path):
    base = Agent(path, 0)

    base.train("jellybeans_dataset")
    print("CNN estimates test.png at {0}. Real value is 345".format(base.eval("test.png")))

    base.save()

def eval(folder, res_json):
    n = Network(4, folder)
    n.eval("test.png", res_json)

def reinforce_network(folder, img, res):
    n = Network(4, folder)
    res_path = "res.json"
    n.eval(img, res_path)
    n.reinforce_all(res_path, img)
    n.eval(img, res)

def reinforce_control(folder, img, res):
    n = Control(folder)
    res_path = "res.json"
    n.eval(img, res_path)
    n.reinforce_all(res_path, img)
    n.eval(img, res)

def control_trial(base, init):
    base = Agent(base, 0)
    base.load()
    network_path = "models"
    test_img_path = "test.png"

    n = Control.gen(n=40, base=base, folder=network_path)

    n.randomize()
    
    print("Evaluating round 0")
    n.eval(test_img_path, init)

def network_trial(base, init):
    base = Agent(base, 0)
    base.load()
    network_path = "models"
    test_img_path = "test.png"

    n = Network.gen(n=40, d=4, base=base, folder=network_path)

    n.randomize()
    
    print("Evaluating round 0")
    n.eval(test_img_path, init)

def analyze(results):
    rounds = ["eval0.json", "eval1.json", "eval2.json", "eval3.json"]
    _analyze(rounds, results)

if __name__ == "__main__":
    if sys.argv[1] == "base":
        base(sys.argv[2])
    elif sys.argv[1] == "network":
        network(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "control":
        control(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "eval":
        eval(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "rn":
        reinforce_network(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "rc":
        reinforce_control(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "nt":
        network_trial(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "ct":
        control_trial(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2])