import networkx as nx
from typing import Dict, List, Any, Optional

class InMemoryKnowledgeGraph:
    """In-memory Knowledge Graph powered by NetworkX."""

    def __init__(self):
        # MultiDiGraph supports multiple directed edges between the same nodes
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity_id: str, label: str, properties: Optional[Dict[str, Any]] = None):
        """Add a node with a label (e.g., Paper, Method, Concept) and properties."""
        props = properties or {}
        props["label"] = label
        props["name"] = props.get("name", entity_id)
        self.graph.add_node(entity_id, **props)

    def add_relation(self, source_id: str, target_id: str, relation_type: str, properties: Optional[Dict[str, Any]] = None):
        """Add a directed relationship between two entities."""
        props = properties or {}
        props["relation"] = relation_type
        self.graph.add_edge(source_id, target_id, key=relation_type, **props)

    def resolve_id(self, node_str: str) -> Optional[str]:
        """Resolve a string name or partial ID to an exact node ID in the graph."""
        if node_str in self.graph:
            return node_str
        clean_query = node_str.replace(' ', '').lower()
        for n, data in self.graph.nodes(data=True):
            clean_name = data.get("name", "").replace(' ', '').lower()
            if clean_name == clean_query or n.lower().endswith(clean_query):
                return n
        return None

    def get_ego_subgraph(self, center_node: str, radius: int = 2) -> nx.MultiDiGraph:
        """Extract a k-hop localized subgraph around a specific node for focused inspection."""
        if center_node not in self.graph:
            return nx.MultiDiGraph()
        
        # Convert to undirected temporarily for radius discovery
        undirected = self.graph.to_undirected()
        subgraph_nodes = nx.single_source_shortest_path_length(undirected, center_node, cutoff=radius).keys()
        return self.graph.subgraph(subgraph_nodes).copy()

    def find_shortest_path(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """Find the shortest path between two concepts or domains."""
        try:
            path = nx.shortest_path(self.graph.to_undirected(), source=source_id, target=target_id)
            return [{"node": node, **self.graph.nodes[node]} for node in path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def calculate_adamic_adar(self, node_a: str, node_b: str) -> float:
        """Calculate the Adamic-Adar score for in-memory link prediction."""
        if node_a not in self.graph or node_b not in self.graph:
            return 0.0
        
        # nx.adamic_adar_index does not support MultiGraph. Cast to simple Graph.
        undirected = nx.Graph(self.graph)
        if undirected.has_edge(node_a, node_b):
            return 0.0  # Already connected
            
        try:
            preds = nx.adamic_adar_index(undirected, [(node_a, node_b)])
            for _, _, score in preds:
                return float(score)
        except Exception:
            return 0.0
        return 0.0
