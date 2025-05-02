from agent import Agent
from network import Network
from control import Control
from tqdm import tqdm
from joblib import Parallel, delayed
from analysis import analyze

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

def control_trial(results, base):
    base = Agent(base, 0)
    base.load()
    network_path = "models"
    test_img_path = "test.png"

    rounds = ["eval0.json", "eval1.json", "eval2.json", "eval3.json"]

    n = Control.gen(n=40, base=base, folder=network_path)

    for a in tqdm(n.agents.values(), desc="Randomizing Models"):
        a.random_train()

    n.eval(test_img_path, rounds[0])

    n.reinforce_all(rounds[0], test_img_path)
    n.eval(test_img_path, rounds[1])

    n.reinforce_all(rounds[1], test_img_path)
    n.eval(test_img_path, rounds[2])

    n.reinforce_all(rounds[2], test_img_path)
    n.eval(test_img_path, rounds[3])

    analyze(rounds, results)

def network_trial(results, base):
    base = Agent(base, 0)
    base.load()
    network_path = "models"
    test_img_path = "test.png"

    rounds = ["eval0.json", "eval1.json", "eval2.json", "eval3.json"]

    n = Network.gen(n=40, d=4, base=base, folder=network_path)

    n.randomize()
    
    print("Evaluating round 0")
    n.eval(test_img_path, rounds[0])

    print("Reinforcing round 1")
    n.reinforce_all(rounds[0], test_img_path)
    print("Evaluating round 1")
    n.eval(test_img_path, rounds[1])

    print("Reinforcing round 2")
    n.reinforce_all(rounds[1], test_img_path)
    print("Evaluating round 2")
    n.eval(test_img_path, rounds[2])

    print("Reinforcing round 3")
    n.reinforce_all(rounds[2], test_img_path)
    print("Evaluating round 3_")
    n.eval(test_img_path, rounds[3])

    analyze(rounds, results)

if __name__ == "__main__":
    if sys.argv[1] == "base":
        base(sys.argv[2])
    if sys.argv[1] == "network":
        network(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "control":
        control(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "eval":
        eval(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "rn":
        reinforce_network(sys.argv[2], sys.argv[3], sys.argv[4])
    if sys.argv[1] == "rc":
        reinforce_control(sys.argv[2], sys.argv[3], sys.argv[4])
    if sys.argv[1] == "nt":
        network_trial(sys.argv[2], sys.argv[3])