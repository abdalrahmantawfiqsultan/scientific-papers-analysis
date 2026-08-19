from src.graph.in_memory_store import InMemoryKnowledgeGraph

def test_resolve_id_exact_and_fuzzy():
    kg = InMemoryKnowledgeGraph()
    
    # Add nodes
    kg.graph.add_node("Domain:1", name="Computer Vision")
    kg.graph.add_node("Domain:2", name="Structural Biology")
    
    # Exact match by ID
    assert kg.resolve_id("Domain:1") == "Domain:1"
    
    # Exact match by Name
    assert kg.resolve_id("Computer Vision") == "Domain:1"
    
    # Fuzzy match by Name (lowercase)
    assert kg.resolve_id("computer vision") == "Domain:1"
    
    # Fuzzy match by partial ID suffix
    assert kg.resolve_id("StructuralBiology") == "Domain:2"
    
    # No match
    assert kg.resolve_id("Unknown Domain") is None
