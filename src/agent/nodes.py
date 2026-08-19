import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.state import DiscoveryState
from src.tools.grounding import search_concepts_in_memory
from src.tools.document_loader import read_local_paper
from src.tools.discovery import detect_scientific_communities, calculate_adamic_adar
from src.tools.validation import calculate_link_probability, analyze_temporal_trend, find_shortest_path
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List

# Ensure the token is picked up if set in environment
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    from dotenv import load_dotenv
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        max_new_tokens=4096,
        do_sample=False,
        timeout=300,
        huggingfacehub_api_token=hf_token
    )
    _llm_instance = ChatHuggingFace(llm=llm_endpoint)
    return _llm_instance

class ExpandedQuery(BaseModel):
    core_concepts: List[str] = Field(description="Core scientific concepts extracted from the query")
    synonyms_and_jargon: List[str] = Field(description="Formal technical terms or domain-specific synonyms")
    target_domains: List[str] = Field(description="Broad scientific domains involved")

def phase_1_grounding(state: DiscoveryState) -> DiscoveryState:
    """Uses LLM query expansion and hybrid search to find exact database names for the user's concepts."""
    query = state["user_query"]
    kg = state.get("kg_reference")
    print(f"--- PHASE 1: Grounding concepts for '{query}' ---")
    
    # 1. Expand the query using Qwen 2.5
    prompt = f"""
You are an expert scientific ontology extractor. Parse this query and expand it into formal scientific search terms: {query}

Output your response strictly as a JSON object matching this schema, with no markdown formatting:
{{
    "core_concepts": ["list of strings"],
    "synonyms_and_jargon": ["list of strings"],
    "target_domains": ["list of strings"]
}}
"""
    
    try:
        raw_response = get_llm().invoke(prompt).content
        import json
        raw_clean = raw_response.strip()
        if raw_clean.startswith("```json"):
            raw_clean = raw_clean[7:]
        elif raw_clean.startswith("```"):
            raw_clean = raw_clean[3:]
        if raw_clean.endswith("```"):
            raw_clean = raw_clean[:-3]
            
        parsed_json = json.loads(raw_clean.strip())
        expanded = ExpandedQuery(**parsed_json)
        search_terms = expanded.core_concepts + expanded.synonyms_and_jargon + expanded.target_domains
        st_query = " ".join(search_terms)
    except Exception as e:
        print(f"Query expansion failed: {e}")
        st_query = query
    
    @tool
    def search_concepts(query: str, top_k: int = 5) -> str:
        """
        Use this tool to find scientific concepts in the database based on semantic meaning and keywords.
        
        Args:
            query: The scientific concept to search for (e.g., 'neural networks').
            top_k: Number of results to return (default 5).
            
        Returns:
            A JSON string containing the exact names of the matched concepts.
        """
        if kg:
            return search_concepts_in_memory(kg, query, top_k)
        return json.dumps([query])
        
    llm = get_llm()
    # We bind the grounding tool and the document loader to the LLM
    llm_with_tools = llm.bind_tools([search_concepts, read_local_paper])
    
    prompt_tool = f"Use the search_concepts tool to find the exact database concept names relevant to this expanded scientific query: {st_query}. If the query asks to read a local paper, use the read_local_paper tool first."
    response = llm_with_tools.invoke([HumanMessage(content=prompt_tool)])

    grounded_concepts = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_concepts":
                tool_result = search_concepts.invoke(tool_call["args"])
                summary_prompt = f"Based on this tool result, list the top 2-3 most relevant exact concept names as a comma-separated list. Result: {tool_result}"
                summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
                grounded_concepts.extend([c.strip() for c in summary.split(',')])
            elif tool_call["name"] == "read_local_paper":
                tool_result = read_local_paper.invoke(tool_call["args"])
                summary_prompt = f"Based on this local paper content, list the top 2-3 most relevant scientific concepts or domains as a comma-separated list. Content snippet: {tool_result}"
                summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
                grounded_concepts.extend([c.strip() for c in summary.split(',')])
    else:
        grounded_concepts = [query]
        
    grounded_concepts = list(set(grounded_concepts))
    return {"grounded_concepts": grounded_concepts}

def compute_soft_bipartite_similarity(concepts_a: List[str], concepts_b: List[str], emb_a=None, emb_b=None) -> tuple[float, List[dict]]:
    """
    Computes bidirectional maximum alignment between two sets of concepts.
    Returns the overall similarity score and the strongest concept-to-concept bridges.
    """
    if not concepts_a or not concepts_b:
        return 0.0, []

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    if emb_a is None or emb_b is None:
        from src.tools.grounding import get_embedder
        embedder = get_embedder()
        emb_a = embedder.encode(concepts_a, normalize_embeddings=True)
        emb_b = embedder.encode(concepts_b, normalize_embeddings=True)

    # Compute pairwise cosine similarity matrix [len(A) x len(B)]
    sim_matrix = cosine_similarity(emb_a, emb_b)

    # Best matches from A -> B and B -> A
    best_match_for_a = np.max(sim_matrix, axis=1)
    best_match_for_b = np.max(sim_matrix, axis=0)

    # Bidirectional Soft-Jaccard score
    soft_jaccard = float((np.sum(best_match_for_a) + np.sum(best_match_for_b)) / (len(concepts_a) + len(concepts_b)))

    # Identify the top matching concept pairs
    top_pairs = []
    for i, row in enumerate(sim_matrix):
        for j, score in enumerate(row):
            if score >= 0.58:  # Calibrated concept-level threshold
                top_pairs.append({
                    "concept_a": concepts_a[i],
                    "concept_b": concepts_b[j],
                    "similarity": float(score)
                })

    top_pairs = sorted(top_pairs, key=lambda x: x["similarity"], reverse=True)[:3]
    return soft_jaccard, top_pairs

def phase_2_structural_discovery(state: DiscoveryState) -> DiscoveryState:
    """Uses topological tools (Community Detection and Adamic-Adar) to find hidden structural bridges."""
    concepts = state.get("grounded_concepts", [])
    kg = state.get("kg_reference")
    print(f"--- PHASE 2: Discovering structural bridges between {concepts} ---")
    
    candidate_bridges = []
    
    # Only attempt topological LLM tools if we have at least 2 grounded concepts
    if len(concepts) >= 2:
        domain_a = concepts[0]
        domain_b = concepts[-1]
    if kg is not None:
        print(f"\n--- [DISCOVERY] Equipping LLM with GraphRAG tools ---")
        
        class AdamicAdarInput(BaseModel):
            node_a: str = Field(description="The exact name of the first scientific concept or paper.")
            node_b: str = Field(description="The exact name of the second scientific concept or paper.")

        @tool(args_schema=AdamicAdarInput)
        def tool_calculate_adamic_adar(node_a: str, node_b: str) -> str:
            """
            Calculates the Adamic-Adar probability index between two disconnected nodes.
            Use this to mathematically validate if two concepts share highly specific structural connections.
            """
            if kg:
                score = calculate_adamic_adar(kg, node_a, node_b)
                return f"Adamic-Adar probability score for {node_a} ↔ {node_b} is {score:.4f}"
            return "Score is 0.0 (Graph not available)."

        class CommunityDetectionInput(BaseModel):
            pass

        @tool(args_schema=CommunityDetectionInput)
        def tool_detect_scientific_communities() -> str:
            """
            Runs the Leiden algorithm to discover and cluster the graph into distinct scientific communities.
            Call this when you need a global overview of how the research fields are divided.
            """
            if kg:
                communities = detect_scientific_communities(kg)
                result_lines = ["Discovered Scientific Communities:"]
                for i, cluster in enumerate(communities):
                    # Truncate to top 5 nodes
                    top_nodes = cluster[:5]
                    result_lines.append(f"Community {i+1} (Total size: {len(cluster)}): {', '.join(str(n) for n in top_nodes)}")
                return "\n".join(result_lines)
            return "Graph is not available."

        llm = get_llm()
        llm_with_tools = llm.bind_tools([
            tool_calculate_adamic_adar, 
            tool_detect_scientific_communities
        ])
        
        prompt = f"""
You are an autonomous GraphRAG scientific agent. You need to uncover hidden structural bridges between '{domain_a}' and '{domain_b}'.

1. First, call `tool_detect_scientific_communities` to survey the global landscape and understand how the fields are clustered.
2. Next, identify potential specific concepts/papers that might bridge the gap.
3. Call `tool_calculate_adamic_adar` on specific pairs of concepts to verify if they share highly specific structural connections.

Execute your tools to synthesize a hypothesis.
"""
        try:
            response = llm_with_tools.invoke([HumanMessage(content=prompt)])
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_func = {
                        "tool_calculate_adamic_adar": tool_calculate_adamic_adar,
                        "tool_detect_scientific_communities": tool_detect_scientific_communities
                    }.get(tool_call["name"])
                    
                    if tool_func:
                        res = tool_func.invoke(tool_call["args"])
                        candidate_bridges.append({"tool": tool_call["name"], "result": res})
        except Exception as e:
            print(f"[PHASE 2] GraphRAG tools failed: {e}")
                    
    # Programmatic Semantic Fallback: Bipartite Matching + Cross-Encoder Verification
    if len(candidate_bridges) == 0 and kg is not None:
        print("\n--- [DISCOVERY] Analyzing papers with Bipartite Matching ---")
        from src.tools.grounding import get_cross_encoder, get_embedder
        
        cross_encoder = get_cross_encoder()
        embedder = get_embedder()
        
        papers = [n for n, d in kg.graph.nodes(data=True) if d.get("label") == "Paper"]
        
        if len(papers) >= 2:
            # Precompute embeddings
            paper_concepts = {}
            paper_descs = {}
            for p in papers:
                neighbors = set()
                for _, v in kg.graph.out_edges(p):
                    if kg.graph.nodes[v].get("label") in ["Concept", "Method", "Dataset"]:
                        neighbors.add(v)
                for u, _ in kg.graph.in_edges(p):
                    if kg.graph.nodes[u].get("label") in ["Concept", "Method", "Dataset"]:
                        neighbors.add(u)
                        
                c_names = [kg.graph.nodes[n].get("name", n) for n in neighbors] or [kg.graph.nodes[p].get("name", p)]
                c_embs = embedder.encode(c_names, normalize_embeddings=True)
                paper_concepts[p] = {"names": c_names, "embs": c_embs}
                paper_descs[p] = kg.graph.nodes[p].get("description", kg.graph.nodes[p].get("name", p))

            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    p1, p2 = papers[i], papers[j]
                    
                    if kg.graph.has_edge(p1, p2) or kg.graph.has_edge(p2, p1):
                        continue
                        
                    c1_names = paper_concepts[p1]["names"]
                    c1_embs = paper_concepts[p1]["embs"]
                    c2_names = paper_concepts[p2]["names"]
                    c2_embs = paper_concepts[p2]["embs"]

                    # Compute Soft-Bipartite Alignment
                    alignment_score, top_bridges = compute_soft_bipartite_similarity(c1_names, c2_names, emb_a=c1_embs, emb_b=c2_embs)
                    
                    # Cross-Encoder Verification on Paper Titles/Descriptions
                    p1_desc = paper_descs[p1]
                    p2_desc = paper_descs[p2]
                    
                    # Predict joint relevance using the Cross-Encoder (STS-B outputs 0-5)
                    raw_cross = float(cross_encoder.predict([(p1_desc, p2_desc)])[0])
                    cross_score = max(0.0, min(1.0, raw_cross / 5.0))  # Normalize to 0-1

                    # Composite Link Discovery Score
                    final_discovery_score = (0.6 * alignment_score) + (0.4 * cross_score)

                    print(f"[LINK CANDIDATE] {p1} <--> {p2} | Score: {final_discovery_score:.3f}")

                    # Inject strong bridges if score passes threshold
                    if final_discovery_score >= 0.55:
                        bridge_info = {
                            "type": "SEMANTIC_BRIDGE",
                            "node_a": p1,
                            "node_b": p2,
                            "name": f"{kg.graph.nodes[p1].get('name')} ↔ {kg.graph.nodes[p2].get('name')}",
                            "similarity_score": round(final_discovery_score, 4),
                            "aligned_concepts": top_bridges,
                            "rationale": (f"Conceptual alignment ({final_discovery_score:.2f}) via: " + 
                                         ", ".join([f"{b['concept_a']} ↔ {b['concept_b']}" for b in top_bridges[:2]]))
                                         if top_bridges else f"Direct semantic alignment ({final_discovery_score:.2f}) between paper descriptions."
                        }
                        candidate_bridges.append(bridge_info)

                        # Inject the link into the visual graph immediately
                        kg.add_relation(
                            p1,
                            p2,
                            "AGENT_DISCOVERY",
                            {
                                "rationale": bridge_info["rationale"],
                                "score": bridge_info["similarity_score"],
                                "top_bridge": top_bridges[0]["concept_a"] + " ↔ " + top_bridges[0]["concept_b"] if top_bridges else "Direct Semantic Alignment"
                            }
                        )

        print(f"--- [DEBUG] END SEMANTIC SEARCH. Found {len(candidate_bridges)} bridges. ---")

    print(f"Discovered {len(candidate_bridges)} semantic/structural bridges.")
    return {"candidate_bridges": candidate_bridges}

def phase_3_validation(state: DiscoveryState) -> DiscoveryState:
    """Calculates mathematical probability and temporal trends for candidates."""
    candidates = state.get("selected_bridges") or state.get("candidate_bridges", [])
    kg = state.get("kg_reference")
    print(f"--- PHASE 3: Validating {len(candidates)} candidate bridges mathematically ---")
    
    @tool
    def tool_calculate_link_probability(node_a: str, node_b: str) -> str:
        """
        Calculates the probability of a hidden link between two nodes.
        
        Args:
            node_a: The name of the first node.
            node_b: The name of the second node.
            
        Returns:
            A JSON string containing Adamic-Adar score and shared neighbors count.
        """
        if kg: return calculate_link_probability(kg, node_a, node_b)
        return "[]"
        
    @tool
    def tool_analyze_temporal_trend(method: str, domain: str) -> str:
        """
        Analyzes how many papers use a specific method in a specific domain grouped by year.
        
        Args:
            method: The method to track.
            domain: The scientific domain.
            
        Returns:
            A JSON string of paper counts per year.
        """
        if kg: return analyze_temporal_trend(kg, method, domain)
        return "[]"
        
    @tool
    def tool_find_shortest_path(concept_a: str, concept_b: str) -> str:
        """
        Finds how two concepts are connected through papers, methods, or authors.
        
        Args:
            concept_a: First concept.
            concept_b: Second concept.
            
        Returns:
            A JSON string of path nodes.
        """
        if kg: return find_shortest_path(kg, concept_a, concept_b)
        return "[]"
        
    llm = get_llm()
    validated_links = []
    llm_with_tools = llm.bind_tools([tool_calculate_link_probability, tool_analyze_temporal_trend, tool_find_shortest_path])
    
    for candidate in candidates[:3]: # Limit to top 3 to save time/tokens
        prompt = f"Validate this candidate bridge: {candidate}. Use tool_calculate_link_probability and tool_analyze_temporal_trend."
        response = llm_with_tools.invoke([HumanMessage(content=prompt)])
        
        validation_data = {"bridge": candidate, "evidence": []}
        if response.tool_calls:
             for tool_call in response.tool_calls:
                 tool_func = {
                    "tool_calculate_link_probability": tool_calculate_link_probability,
                    "tool_analyze_temporal_trend": tool_analyze_temporal_trend,
                    "tool_find_shortest_path": tool_find_shortest_path
                 }.get(tool_call["name"])
                 
                 if tool_func:
                     res = tool_func.invoke(tool_call["args"])
                     validation_data["evidence"].append(res)
                     
        validated_links.append(validation_data)
        
    return {"validated_links": validated_links}

def phase_3_5_self_correction(state: DiscoveryState) -> DiscoveryState:
    """Uses Qwen's reasoning to filter out false positive bridges."""
    validated_links = state.get("validated_links", [])
    print(f"--- PHASE 3.5: Self-Correcting and Reviewing {len(validated_links)} links ---")
    
    if not validated_links:
        return {"validated_links": []}
        
    critique_prompt = f"""
    You are a rigorous peer reviewer. Review these candidate graph bridges:
    {validated_links}
    
    Task: Identify any links that are merely coincidental or too generic to be a meaningful scientific discovery. 
    Return only the high-value, genuinely non-obvious methodological bridges as JSON format without markdown.
    If none are good, return an empty JSON list [].
    Ensure your output is strictly a JSON list of the valid links.
    """
    
    response = get_llm().invoke([HumanMessage(content=critique_prompt)]).content
    
    try:
        filtered_links = json.loads(response)
        if isinstance(filtered_links, list):
            return {"validated_links": filtered_links}
    except Exception as e:
        print(f"Self-correction parsing failed: {e}")
        
    # Fallback to the original links if JSON parsing fails
    return {"validated_links": validated_links}

def phase_4_synthesis(state: DiscoveryState) -> DiscoveryState:
    """LLM writes the final scientific hypothesis using the validated evidence."""
    print("--- PHASE 4: Synthesizing final hypothesis ---")
    
    validated_data = state.get("validated_links", [])
    
    prompt = f"""
    You are the HSLDE Engine. 
    Look at the candidate bridges discovered below. YOU MUST write your hypothesis about these exact bridges!
    
    Discovered Bridges:
    {state.get("candidate_bridges", [])}
    
    Validated Links (if any):
    {json.dumps(validated_data)}
    
    Write a 3-sentence scientific hypothesis explaining why these concepts/papers are highly similar.
    """
    
    response = get_llm().invoke([SystemMessage(content=prompt)])
    
    return {"final_hypothesis": response.content}
