import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import string

def generate_graph_data(n):
    """Generate graph data including relationships and initial positions for a fully connected graph with alphabetical node names."""
    # Ensure we do not exceed 26 nodes for alphabets
    if n > 26:
        raise ValueError("Maximum number of nodes supported is 26 (A-Z).")
    
    # Create a fully connected graph with alphabetical node names
    nodes = list(string.ascii_uppercase[:n])
    G = nx.complete_graph(nodes)
    
    # Generate weights from a normal distribution and clip them to be within [-10, 10]
    num_edges = len(G.edges())
    weights_array = np.random.normal(loc=0, scale=5, size=num_edges)
    weights_clipped = np.clip(weights_array, -10, 10).astype(int)

    # Assign clipped weights to edges
    for edge, weight in zip(G.edges(), weights_clipped):
        G.edges[edge]['weight'] = weight

    # Get initial positions in a circular layout and adjust position of 'A'
    positions = nx.circular_layout(G)
    angle_offset = np.angle(positions['A'][0] + 1j*positions['A'][1])
    positions = {node: ((pos[0] * np.cos(-angle_offset) - pos[1] * np.sin(-angle_offset)),
                        (pos[0] * np.sin(-angle_offset) + pos[1] * np.cos(-angle_offset)))
                 for node, pos in positions.items()}
    positions['A'] = (0, 0)  # Set 'A' to position (0,0)

    # Create the relationships dictionary
    relationships = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
    
    # Adjust positions to dictionary
    initial_positions = {node: tuple(pos) for node, pos in positions.items()}

    return relationships, initial_positions, G
    # return {
    #     "relationships": relationships,
    #     "initial_positions": initial_positions
    # }, G

# Example usage:
# n = 14  # Number of nodes
# graph_data, G = generate_graph_data(n)
# print("Relationships:", graph_data["relationships"])
# print("Initial Positions:", graph_data["initial_positions"])
#
# nx.draw(G, pos=graph_data["initial_positions"], with_labels = True)