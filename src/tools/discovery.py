import networkx as nx
from cdlib import algorithms

def detect_scientific_communities(kg):
    """Uses the Leiden algorithm to find distinct research clusters."""
    if len(kg.graph.nodes) == 0:
        return []
    
    # Leiden algorithm using cdlib
    try:
        communities = algorithms.leiden(kg.graph)
        return communities.communities
    except Exception as e:
        print(f"Community detection failed: {e}")
        return []

def calculate_adamic_adar(kg, node_a: str, node_b: str) -> float:
    """Calculates the topological probability of a hidden connection."""
    # Adamic-Adar only works for undirected simple graphs in NetworkX
    undirected_graph = nx.Graph(kg.graph)
    
    if not undirected_graph.has_node(node_a) or not undirected_graph.has_node(node_b):
        return 0.0
        
    try:
        preds = nx.adamic_adar_index(undirected_graph, [(node_a, node_b)])
        for u, v, p in preds:
            return p
    except nx.NetworkXError as e:
        # e.g. nodes are not in graph
        return 0.0
    return 0.0
