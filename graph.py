import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

edges = []
size = 40
# for i in range(size):
#     for j in [-2, -1, 1, 2]:
#         edges.append(((i + size + j) % size, (i)))

G.add_edges_from(edges)
G.add_nodes_from(range(40))

# Generate circular positions
pos = nx.spiral_layout(G, equidistant=True, resolution=1)

# Draw with circular layout
nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=100)
plt.savefig("graph_circular.png")  # Save as image
plt.show()