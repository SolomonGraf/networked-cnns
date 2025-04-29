import os, re, json
from tqdm import tqdm
from agent import Agent

class Control:
    """__init__
    
    Keyword arguments:
    n -- size of network
    d -- constant degree of egal. network - must be even
    Return: returns object
    """

    def __init__(self, folder: str):
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

    @staticmethod
    def gen(n, base, folder):
        for i in tqdm(range(n), desc="Populating Models"):
            base.deepcopy(i, os.path.join(folder, f"model_{i}.pth"))
        
        return Control(folder)
    
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
        with open(res_path, "r") as f:
            res = json.load(f)
        for i, a in tqdm(self.agents.items(), desc="Reinforcing"):
            a.reinforce(img, res[str(i)])
            a.save()