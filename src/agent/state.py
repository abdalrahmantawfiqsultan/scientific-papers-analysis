from typing import TypedDict, List, Dict, Any

class DiscoveryState(TypedDict):
    """The State acts as the shared memory for the LangGraph workflow."""
    user_query: str
    kg_reference: Any                        # Pass the InMemoryKnowledgeGraph here
    grounded_concepts: List[str]             # Filled in Phase 1 (Grounding)
    candidate_bridges: List[Dict[str, Any]]  # Filled in Phase 2 (Structural Discovery)
    selected_bridges: List[Dict[str, Any]]   # Filtered by Human-in-the-Loop before Phase 3
    validated_links: List[Dict[str, Any]]    # Filled in Phase 3 (Validation)
    final_hypothesis: str                    # Filled in Phase 4 (Synthesis)
