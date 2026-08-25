import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional

class VectorStore:
    """Decoupled vector storage and computation for nodes in the graph."""
    def __init__(self, kg: "InMemoryKnowledgeGraph"):
        self._kg = kg
        self._cache: Dict[str, np.ndarray] = {}

    def get_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Fetch embeddings for node_ids. Computes and caches missing embeddings."""
        if not node_ids:
            return np.array([])
            
        missing_ids = [nid for nid in node_ids if nid not in self._cache]
        
        if missing_ids:
            # Lazy import to avoid circular dependency
            from src.tools.grounding import get_embedder
            texts = [self._kg._get_embedding_text(nid) for nid in missing_ids]
            
            if texts:
                new_embs = get_embedder().encode(texts)
                for nid, emb in zip(missing_ids, new_embs):
                    self._cache[nid] = emb
                    
        return np.array([self._cache[nid] for nid in node_ids])
        
    def evict(self, node_id: str) -> None:
        """Remove a node's embedding from the cache."""
        self._cache.pop(node_id, None)

class InMemoryKnowledgeGraph:
    """In-memory Knowledge Graph powered by NetworkX."""

    def __init__(self):
        # MultiDiGraph supports multiple directed edges between the same nodes
        self.graph = nx.MultiDiGraph()
        self.vectors = VectorStore(self)
        
    def _get_embedding_text(self, node_id: str) -> str:
        """Determine the canonical text representation of a node for semantic embedding."""
        if not self.graph.has_node(node_id):
            return ""
        data = self.graph.nodes[node_id]
        return data.get("description") or data.get("name", node_id)

    def add_entity(self, entity_id: str, label: str, properties: Optional[Dict[str, Any]] = None):
        """Add a node with a label (e.g., Paper, Method, Concept) and properties."""
        props = properties or {}
        props["label"] = label
        props["name"] = props.get("name", entity_id)
        self.graph.add_node(entity_id, **props)

    _VALID_EDGE_SOURCES = {"extracted", "algorithm", "citation", "user"}

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        source: str,
        confidence: float,
        provenance: Dict[str, Any],
        properties: Optional[Dict[str, Any]] = None,
    ):
        """Add a directed relationship between two entities with mandatory provenance tagging."""
        if source not in self._VALID_EDGE_SOURCES:
            raise ValueError(f"edge source must be one of {self._VALID_EDGE_SOURCES}, got {source!r}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")
            
        props = properties or {}
        props.update({"relation": relation_type, "source": source, "confidence": confidence, "provenance": provenance})
        self.graph.add_edge(source_id, target_id, key=relation_type, **props)

    def delete_entity(self, node_id: str):
        """Removes a node and all of its incident edges from the graph."""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            self.vectors.evict(node_id)

    def edit_entity(self, node_id: str, new_properties: Dict[str, Any]):
        """Updates a node's properties, coercing known types, and conditionally evicts its vector cache."""
        if not self.graph.has_node(node_id):
            return
            
        old_text = self._get_embedding_text(node_id)
            
        # Type coercion for known fields to prevent type fragmentation
        if "year" in new_properties and new_properties["year"] is not None:
            try:
                new_properties["year"] = int(new_properties["year"])
            except ValueError:
                pass
                
        for key, value in new_properties.items():
            self.graph.nodes[node_id][key] = value
            
        # Conditionally evict the cache if the semantic text changed
        if self._get_embedding_text(node_id) != old_text:
            self.vectors.evict(node_id)

    def delete_relation(self, source_id: str, target_id: str, relation_type: str):
        """Deletes a specific edge by its source, target, and relation type."""
        if self.graph.has_edge(source_id, target_id, key=relation_type):
            self.graph.remove_edge(source_id, target_id, key=relation_type)
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
