import json

def load_file(path):
    res = []
    with open(path, "r") as f:
        res = json.load(f)
    return res

base = load_file("base.json")
r1 = load_file("round1.json")
r2 = load_file("round2.json")
r3 = load_file("round3.json")

results = {id : [] for id in base.keys()}

mad = sum([abs(v - 345) for v in list(base.values())])/len(base.keys())
print(mad)
for k, v in base.items():
    results[k].append(v)

mad = sum([abs(v - 345) for v in list(r1.values())])/len(base.keys())
print(mad)
for k, v in r1.items():
    results[k].append(v)

mad = sum([abs(v - 345) for v in list(r2.values())])/len(base.keys())
print(mad)
for k, v in r2.items():
    results[k].append(v)

mad = sum([abs(v - 345) for v in list(r3.values())])/len(base.keys())
print(mad)
for k, v in r3.items():
    results[k].append(v)

with open("control_results.json", "w") as f:
    json.dump(results, f, indent=2)