import os, re, json
from tqdm import tqdm
from agent import Agent
from joblib import Parallel, delayed

class Network:
    """__init__
    
    Keyword arguments:
    n -- size of network
    d -- constant degree of egal. network - must be even
    Return: returns object
    """

    def __init__(self, d: int, folder: str):
        self.agents: dict[int, Agent]= {}
        model_files = [f for f in os.listdir(folder) 
                      if f.endswith('.pth') and re.match(r'model_\d+\.pth', f)]
        
        if not model_files:
            raise ValueError(f"No valid model files found in {folder}")
        
        for model_file in tqdm(model_files, desc="Loading models"):
            try:
                # Extract ID from filename (e.g., "model_5.pth" -> 5)
                model_id = int(re.search(r'model_(\d+)\.pth', model_file).group(1))
                
                # Create agent and load weights
                agent = Agent(os.path.join(folder, model_file), model_id)
                agent.load()
                
                self.agents[model_id] = agent
            except Exception as e:
                print(f"Failed to load {model_file}: {str(e)}")
                continue
        
        self.size = len(model_files)
        assert d % 2 == 0
        self.deg = d

    @staticmethod
    def gen(n, d, base, folder):
        for i in tqdm(range(n), desc="Populating Models"):
            base.deepcopy(i, os.path.join(folder, f"model_{i}.pth"))
        
        return Network(d, folder)

    """get_neighbors
    
    Keyword arguments:
    i -- index of agent
    Return: indices of neighbors in the network
    """
    
    def get_neighbors(self, i):
        neighbors = [i + j for j in range(-self.deg//2, self.deg//2 + 1)]
        neighbors = map(lambda x : (x + self.size) % self.size, neighbors)
        return neighbors
    
    def eval(self, img, res_json):
        results = [0] * self.size

        for id, a in self.agents.items():
            r = a.eval(img)
            results[id] = r
            with open(res_json, 'w') as file:
                json.dump({str(id) : i for id, i in enumerate(results)}, file, indent=2)

    def get_avg(self, id, results_path):
        ns = self.get_neighbors(id)
        with open(results_path, "r") as f:
            res = json.load(f)
        vals = [res[str(n)] for n in ns]
        return round(sum(vals)/len(vals))
    
    def reinforce_all(self, res_path, img):
        avgs = {id: self.get_avg(int(id), res_path) for id in self.agents.keys()}

        def aux(c, a):
            a.reinforce(img, c)
            a.save()

        with Parallel(n_jobs=-1) as p:
            p(
            delayed(aux) (avgs[i], a) 
            for i, a in tqdm(self.agents.items())
            )