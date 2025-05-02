import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

edges = []
size = 40
for i in range(size):
    for j in [-2, -1, 1, 2]:
        edges.append(((i + size + j) % size, (i)))

G.add_edges_from(edges)
G.add_nodes_from(range(40))

# Generate circular positions
pos = nx.kamada_kawai_layout(G)

# Create node color list - green for nodes divisible by 4, lightblue otherwise
node_colors = ['green' if node % 1 == 0 else 'lightblue' for node in G.nodes()]

# Draw with circular layout and custom node colors
nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=100)
plt.savefig("graph_circular.png")  # Save as image