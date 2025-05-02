import json

def load_file(path):
    res = []
    with open(path, "r") as f:
        res = json.load(f)
    return res

def mad(res):
    return sum([abs(v - 345) for v in list(res.values())])/len(res.keys())

def analyze(rounds, output):
    
    data = [load_file(f) for f in rounds]
    results = {id : [] for id in data[0].keys()}
    for d in data:
        print(mad(d))
        for k, v in d.items():
            results[k].append(v)

    with open(output, "w") as f:
        json.dump(results, f, indent=2)