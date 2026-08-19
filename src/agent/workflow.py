from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import DiscoveryState
from src.agent.nodes import (
    phase_1_grounding,
    phase_2_structural_discovery,
    phase_3_validation,
    phase_3_5_self_correction,
    phase_4_synthesis
)

def build_workflow():
    """Builds and compiles the LangGraph state machine with memory and HITL."""
    
    # Early Exit node
    def early_exit(state: DiscoveryState) -> DiscoveryState:
        """Populates the final hypothesis when no bridges are found."""
        concepts = state.get("grounded_concepts", [])
        concept_names = ", ".join(concepts) if concepts else "these fields"
        
        fallback_msg = (
            f"**Discovery Halted:** The engine explored the topology between `{concept_names}`, "
            f"but found no shared methods, boundary authors, or structural holes. "
            f"They currently appear completely disconnected in the knowledge base."
        )
        return {"final_hypothesis": fallback_msg}

    # Routing function
    def check_bridges_exist(state: DiscoveryState) -> str:
        """Determines whether to proceed to Validation or exit early."""
        bridges = state.get("candidate_bridges", [])
        if len(bridges) == 0:
            return "empty"
        else:
            return "continue"

    # 1. Initialize the StateGraph with our State schema
    workflow = StateGraph(DiscoveryState)
    
    # 2. Register the nodes (our phases)
    workflow.add_node("grounding", phase_1_grounding)
    workflow.add_node("discovery", phase_2_structural_discovery)
    workflow.add_node("validation", phase_3_validation)
    workflow.add_node("self_correction", phase_3_5_self_correction)
    workflow.add_node("synthesis", phase_4_synthesis)
    workflow.add_node("early_exit", early_exit)
    
    # 3. Add the sequential edges to enforce the workflow
    workflow.add_edge(START, "grounding")
    workflow.add_edge("grounding", "discovery")
    
    # Conditional Edge
    workflow.add_conditional_edges(
        "discovery",            # Source node
        check_bridges_exist,    # The routing function
        {
            "continue": "validation", # If returns "continue", go to Phase 3
            "empty": "early_exit"     # If returns "empty", go to Fallback
        }
    )
    
    workflow.add_edge("validation", "self_correction")
    workflow.add_edge("self_correction", "synthesis")
    workflow.add_edge("synthesis", END)
    workflow.add_edge("early_exit", END)
    
    # 4. Initialize MemorySaver for persistent cross-turn memory
    memory = MemorySaver()
    
    # 5. Compile the graph into an executable application
    app = workflow.compile(
        checkpointer=memory
    )
    
    return app

# Expose a ready-to-use app instance
agent_app = build_workflow()
