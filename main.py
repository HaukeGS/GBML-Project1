import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import task_3_sample_data as sample_data
import random
import json

"""
The nodes and edges will be of the form:
nodes = ["A", "B", "C", "D", "E", "F", "G", ...] (list)

edges = [
    ("A", "B"), ("A", "C"),
    ("B", "D"), ("B", "E"), ...
] (list of tuples)
"""


def find_relationships(node, nodes, edges):
    """
    Finds and returns the parents, descendants, and non-descendants of a given node 
    in a directed graph.

    Parameters:
    node (str): The node for which the relationships are to be found.
    nodes (list of str): A list of all nodes in the graph.
    edges (list of tuples): A list of tuples representing directed edges between nodes.

    Returns:
    list: A sorted list of parent nodes.
    list: A sorted list of descendant nodes.
    list: A sorted list of non-descendant nodes.
    """

    def get_descendants(node, edges):
        descendants = []
        for edge in edges:
            if edge[0] == node:
                descendants.append(edge[1])
                descendants += get_descendants(edge[1], edges)
        return descendants

    def get_parents(node, edges):
        parents = []
        for edge in edges:
            if edge[1] == node:
                parents.append(edge[0])
                parents += get_parents(edge[0], edges)
        return parents

    descendants = get_descendants(node, edges)
    non_descendant = list(set(nodes) - set(descendants))
    non_descendant.remove(node)
    return sorted(get_parents(node, edges)), sorted(get_descendants(node, edges)), sorted(non_descendant)


def graph_is_DAG(edges):
    """
    Analyzes a directed graph to determine if it is a Directed Acyclic Graph (DAG).

    Parameters:
    edges (list of tuples): A list of tuples representing directed edges between nodes.

    Returns:
    bool: True if the graph is a DAG, False otherwise.
    """

    def search_for_DAG_recursively(edges, node, nodes_visited, nodes_confirmed):
        nodes_visited.append(node)
        for edge in edges:
            if edge[0] == node:
                if edge[1] in nodes_visited:
                    return False, None
                elif edge[1] in nodes_confirmed:
                    continue
                else:
                    ret, nodes_vis = search_for_DAG_recursively(edges, edge[1], nodes_visited, nodes_confirmed)
                    if ret:
                        nodes_confirmed.update(nodes_vis)
                    else:
                        return False, None
        return True, nodes_visited

    nodes = set()
    nodes_confirmed = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    for node in nodes:
        dag, nodes_visited = search_for_DAG_recursively(edges, node, [], nodes_confirmed)
        if not dag:
            return False
        else:
            nodes_confirmed.update(nodes_visited)
    return True


def get_top_right_and_bottom_left_positions(positions):
    """
    Calculates the top right and bottom left positions that the nodes span.

    @param positions:
    @return:
    """
    top_right = [0, 0]
    bottom_left = [0, 0]
    for node in positions.values():
        top_right[0] = max(top_right[0], node[0])
        top_right[1] = max(top_right[1], node[1])
        bottom_left[0] = min(bottom_left[0], node[0])
        bottom_left[1] = min(bottom_left[1], node[1])
    return top_right, bottom_left


def detect_communities(nodes, positions, relationships, number_nodes_for_core_threshold=None,
                       clusters_distance_scale=None):
    """
    Implements community detection in a social network by integrating spatial and relational data.

    Parameters:
    nodes (list of str): A list of all nodes.
    positions (dict): A dictionary of node positions.
    relationships (dict): A dictionary of relational strengths between nodes.

    Returns:
    dict: A dictionary where each key is a cluster identifier and value is a list of nodes in that cluster.
    """

    def get_cluster_index_for_node(node: str, clusters: dict) -> int | None:
        """
        @param node: The node which is searched for in the clusters
        @param clusters: The clusters which are searched through for the node
        @return: The cluster identifier for the node or None if the node is in no cluster
        """
        for key, cluster in clusters.items():
            if node in cluster:
                return key
        else:
            return None

    def get_relationship(u, v, relationships):
        """
        @param u: Node 1
        @param v: Node 2
        @param relationships: Relationships between all nodes
        @return: The relationship or 0 if the relationship is not found
        """
        relationship = relationships.get((u, v), 0)
        if relationship == 0:
            relationship = relationships.get((v, u), 0)
        return relationship

    def is_in_distance(u, v, positions, threshhold) -> bool:
        """
        @param u: Node 1
        @param v: Node 2
        @param positions: Positions of all nodes
        @param threshhold: The distance that is checked wheter the nodes are within it or not
        @return: True when the nodes are withing threshold distance, False when not
        """
        return np.linalg.norm(np.subtract(positions[u], positions[v])) < threshhold

    def get_distance_threshold(positions, scale=None) -> float:
        """
        Calculate the area spanned by the nodes and take 1/10 of the diagonal as distance threshold

        @param positions: Positions of all nodes
        @return: distance threshold
        @param scale: scale for the distance threshold. The higher the scale, the more clusters will be found. Should be between 0 and 1
        """
        if scale is None:
            scale = 0.1  # default value
        elif scale >= 1:
            raise ValueError("A scale this high will result in one large cluster")
        top_right, bottom_left = get_top_right_and_bottom_left_positions(positions)
        hypothesis = np.linalg.norm(np.subtract(top_right, bottom_left))
        return hypothesis * scale

    # assign distance threshold based on the spatial layout
    distance_threshold = get_distance_threshold(positions, clusters_distance_scale)
    # assign number of friends necessary to become a core node based on the number of nodes present if not passed as a parameter
    if number_nodes_for_core_threshold is None:
        number_nodes_for_core_threshold = np.ceil(len(nodes) / 19)
    friend_counts = dict()
    cores = list()
    noise = set()
    borders = set()
    clusters = dict()

    # count number of nodes in proximity
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if is_in_distance(u, v, positions, distance_threshold):
                friend_counts[u] = friend_counts.get(u, 0) + 1

    # sort for the nodes with the most friends. Then classify the nodes
    friends_counts_sorted = dict(sorted(friend_counts.items(), key=lambda item: item[1], reverse=True))
    for node, friends in friends_counts_sorted.items():
        if friends >= number_nodes_for_core_threshold:
            cores.append(node)
        elif friends == 0:
            noise.add(node)
        else:
            borders.add(node)

    # create clusters with the cores
    for core in cores:
        for node in nodes:
            if core == node:
                continue
            if is_in_distance(core, node, positions, distance_threshold):
                # joins the first cluster they find lexicographically, even if they are multiple suitable candidates.
                index = get_cluster_index_for_node(node, clusters)
                if index != None:
                    clusters[index].append(core)
                    break
        else:
            clusters[len(clusters.keys())] = [core]

    # find the most suitable cluster for the border nodes. If the border node is in reach of two different clusters,
    # the average relationship to each cluster node makes the difference
    for border in borders:
        possible_clusters = []
        for core in cores:
            if core == border:
                raise ValueError("A node can't be a core and a border node at the same time")
            if is_in_distance(border, core, positions, distance_threshold):
                clusters_identifier = get_cluster_index_for_node(core, clusters)
                possible_clusters.append(clusters_identifier)
        if len(possible_clusters) == 1:
            clusters[possible_clusters[0]].append(border)
        elif len(possible_clusters) > 1:
            average_relationships = {}
            for identifier in possible_clusters:
                sum_relationship = 0
                for node in clusters[identifier]:
                    sum_relationship += get_relationship(border, node, relationships)
                average_relationships[identifier] = sum_relationship / len(clusters[identifier])
            key = max(average_relationships, key=average_relationships.get)
            clusters[key].append(border)
        else:
            noise.add(border)
    return clusters


def get_color_for_cluster(identifier: int) -> str:
    """
    Returns a color for a cluster. Since more than 7 clusters are not to be expected, the colors of the rainbow are used.
    If more clusters should be present, they are not colored as the get statement returns the default color of networkx for not applicable identifiers.

    @param identifier: identifier of the cluster
    @return: color as string
    """
    rainbow_colors = {
        0: "#FF0000",  # Red
        1: "#FFA500",  # Orange
        2: "#FFFF00",  # Yellow
        3: "#008000",  # Green
        4: "#0000FF",  # Blue
        5: "#4B0082",  # Indigo
        6: "#EE82EE"  # Violet
    }
    return rainbow_colors.get(identifier, '#1f78b4')


def draw_graph(relationships, positions, number_nodes_for_core_threshold=None, clusters_distance_scale=None):
    """
    @param relationships: edges of two nodes (tupel) as keys and relationship strength (int) as values
    @param positions: nodes (str) as keys and position (tupel) as values
    @param number_nodes_for_core_threshold: Optional core threshold parameter for calculating clusters
    @param clusters_distance_scale: Optional scale for the distance in which nodes are clustered together. Should be between 0 and 1 with 0.1 as default
    """
    G = nx.Graph()
    for node in positions.keys():
        G.add_node(node, pos=positions[node])
    for edge in relationships.keys():
        distance = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
        relationship = relationships[edge]
        color = 'b'
        if relationship > 0:
            color = 'g'
        elif relationship < 0:
            color = 'r'
        G.add_edge(edge[0], edge[1], distance=distance, weight=abs(relationship), color=color)
    fig, ax = plt.subplots()
    nx.draw(G, positions, with_labels=True, ax=ax)

    # edges
    weights = [G[u][v]['weight'] / 2 for u, v in G.edges()]
    colors = [G[u][v]['color'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, positions, edge_color=colors, width=weights, alpha=0.8)

    # labels
    labels = nx.get_edge_attributes(G, 'distance')
    labels = {edge: round(float(label), 2) for edge, label in labels.items()}
    # nx.draw_networkx_edge_labels(G, positions, edge_labels=labels)

    # nodes / clusters
    clusters = detect_communities(list(positions.keys()), positions, relationships, number_nodes_for_core_threshold,
                                  clusters_distance_scale)
    for identifier, cluster in clusters.items():
        color = get_color_for_cluster(identifier)
        p = {node: positions[node] for node in cluster}
        nx.draw_networkx_nodes(G, p, cluster, node_color=color, edgecolors='#000000')

    # axis
    plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()


def check_constraints(relationships: dict, positions: dict, prints=False) -> bool:
    """
    @param relationships: edges of two nodes (tupel) as keys and relationship strength (int) as values
    @param positions: nodes (str) as keys and position (tupel) as values
    @param prints: boolean whether the print statements should be put into the console
    @return: False when constrains are not met, True when they are
    """
    top_right, bottom_left = get_top_right_and_bottom_left_positions(positions)
    if abs(top_right[0] - bottom_left[0]) > 10.0 or abs(top_right[1] - bottom_left[1]) > 10.0:
        if prints:
            print("Violating constraint to keep the area 10x10 for points of top right: ", top_right,
                  " and bottom left: ", bottom_left)
        else:
            return False
    for edge in relationships.keys():
        distance = np.linalg.norm(np.subtract(positions[edge[0]], positions[edge[1]]))
        if relationships[edge] > 0 and distance > 4.0:
            if prints:
                print("Violating constraint to keep edges between friends within 4 for edge ", edge)
            else:
                return False
        if relationships[edge] < 0 and distance < 1.0:
            if prints:
                print("Violating constraint to keep edges between enemies more than 1 for edge ", edge)
            else:
                return False
    return True


def randomize_positions(positions: dict) -> dict:
    for node, pos in positions.items():
        if node != 'A':
            positions[node] = np.array([random.uniform(-5, 5), random.uniform(-5, 5)])
    return positions


def optimize_relationships(relationships, initial_positions, number_nodes_for_core_threshold=None,
                           clusters_distance_scale=None):
    """
    Applies a modified Fruchterman-Reingold algorithm to optimize the placements of nodes
    based on their relational weights in a 10x10 area.

    Parameters:
    relationships (dict): A dictionary where keys are tuples representing the nodes,
                          and values are the weights (-10 to 10) indicating relationship strength.
    initial_positions (dict): A dictionary of node positions with nodes as keys and coordinates as values.

    Returns:
    dict: Final positions of each node.
    """

    def attraction_force(d, k):
        return d ** 2 / k

    def repulsion_force(d, k):
        return -k ** 2 / d

    positions = initial_positions

    C = 1
    area = 10 * 10
    k = C * np.sqrt(area / len(initial_positions.keys()))
    t = 1
    i = 0

    while t > 0:
        n_disp_dict = {node: np.array([0.0, 0.0]) for node in initial_positions.keys()}

        # calculate repulsive forces
        for u in initial_positions.keys():
            n_disp = 0
            for v in initial_positions.keys():
                if u != v:
                    delta = np.subtract(positions[v], positions[u])
                    delta_norm = np.linalg.norm(delta)
                    n_disp += (delta / delta_norm) * repulsion_force(delta_norm, k)
            n_disp_dict[u] += n_disp

        # calculate relationship forces
        for edge in relationships.keys():
            # normal FR attraction forces for edges
            u = edge[0]
            v = edge[1]
            delta = np.subtract(positions[v], positions[u])
            delta_norm = np.linalg.norm(delta)
            disp = (delta / delta_norm) * attraction_force(delta_norm, k)

            # depending on relationship type and distance, enhance or reduce attraction/repulsion forces for relationships
            if relationships[edge] > 0:
                # for friends increase force greatly for friends with distance greater than 4 to try to comply with constraint
                if delta_norm > 4.0:
                    disp = disp * (2 * (delta_norm ** 2))
                else:
                    disp = disp * (np.sqrt(delta_norm))
            elif relationships[edge] < 0:
                # for enemies make force repulsive and increase greatly within short distances, while being significantly lower in a medium distance and zero for long distances
                if delta_norm < 2.0:
                    disp = disp * ((((np.sqrt(25) * delta_norm) / 4) ** 2) - 25)
                elif delta_norm < 4.0:
                    disp = disp * -1
                else:
                    disp = disp * 0
            else:
                disp = disp * 0
            # factor in the strength of the relationship. Full multiplication was too much, so I settlede for a range between 1 and 3
            disp = disp * (abs(relationships[edge] / 5) + 1)

            n_disp_dict[u] += disp
            n_disp_dict[v] -= disp

        for node in positions.keys():
            # skip node A, to ensure it stays at (0, 0). There are forces lost because of this, but I tried adding them to the other nodes and the results were not satisfactory
            if node == "A":
                continue
            n_disp = n_disp_dict[node]
            positions[node] += (n_disp / np.linalg.norm(n_disp)) * min(np.linalg.norm(n_disp), t)

        # reduce temperature. We make 1000 iterations or until we comply with constraints
        t = t - 0.001

        # draw every hundredth graph
        if i % 100 == 0:
            draw_graph(relationships, positions, number_nodes_for_core_threshold, clusters_distance_scale)
        i += 1
        # break if constraints are satisfied
        if check_constraints(relationships, positions):
            break

    # draw the final graph
    draw_graph(relationships, positions, number_nodes_for_core_threshold, clusters_distance_scale)
    if check_constraints(relationships, positions):
        print("Finished optimizing spatial layout for constraints")
    return positions


def read_last_graph() -> dict:
    with open('last_graph.json', 'r') as file:
        data = json.load(file)
        relationships = {tuple(key.split()): np.int32(value) for key, value in data['relationships'].items()}
        initial_positions = data['initial_positions']
        return relationships, initial_positions


# relationships, initial_positions, G = sample_data.generate_graph_data(25)
relationships, initial_positions = read_last_graph()
json_relationships = {" ".join(key): int(value) for key, value in relationships.items()}
with open('last_graph.json', 'w') as file:
    json.dump({'relationships': json_relationships, 'initial_positions': initial_positions}, file)
final_positions = optimize_relationships(relationships, initial_positions)

check_constraints(relationships, final_positions, prints=True)
draw_graph(relationships, final_positions, number_nodes_for_core_threshold=1)
