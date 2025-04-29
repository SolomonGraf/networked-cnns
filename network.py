import os
from tqdm import tqdm

class Network:
    """__init__
    
    Keyword arguments:
    n -- size of network
    d -- constant degree of egal. network - must be even
    Return: returns object
    """

    def __init__(self, n, d, base):
        self.agents = [base.deepcopy(i, os.path.join("models", f"model_{i}.pth")) for i in tqdm(range(n), desc="Populating Models")]
        self.size = n
        assert d % 2 == 0
        self.deg = d

    """get_neighbors
    
    Keyword arguments:
    i -- index of agent
    Return: indices of neighbors in the network
    """
    
    def get_neighbors(self, i):
        neighbors = [i + j for j in range(-self.deg/2, self.deg/2 + 1)]
        neighbors.remove(i)
        neighbors = map(lambda x : (x + self.size) % self.size, neighbors)
        return neighbors
    
    