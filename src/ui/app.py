import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
load_dotenv()
import networkx as nx

from src.graph.in_memory_store import InMemoryKnowledgeGraph
from src.graph.visualizer import GraphVisualizer2D, COLOR_PALETTE
from src.ui.styles import apply_custom_theme
from src.ui.components import render_stepper, render_temporal_chart

st.set_page_config(
    page_title="HSLDE | Scientific Discovery Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS
apply_custom_theme()

import time
from src.agent.workflow import agent_app

# 1. State Management
def get_graph_store():
    """Ensures each session gets its own graph instance."""
    if "kg" not in st.session_state:
        kg = InMemoryKnowledgeGraph()
        # Seed sample cross-disciplinary graph
        kg.add_entity("Field:CV", "Field", {"name": "Computer Vision"})
        kg.add_entity("Field:Bio", "Field", {"name": "Structural Biology"})
        
        kg.add_entity("Method:GNN", "Method", {"name": "Graph Neural Network", "description": "Used for representation learning on graphs."})
        kg.add_entity("Concept:ProtFold", "Concept", {"name": "Protein Folding", "description": "Predicting 3D structure from amino acid sequence."})
        
        kg.add_entity("Paper:P1", "Paper", {"title": "GNNs for Image Graphs", "year": 2023})
        kg.add_entity("Paper:P2", "Paper", {"title": "Deep Learning for Fold Prediction", "year": 2024})
        
        kg.add_relation("Paper:P1", "Field:CV", "BELONGS_TO")
        kg.add_relation("Paper:P1", "Method:GNN", "USES_METHOD")
        
        kg.add_relation("Paper:P2", "Field:Bio", "BELONGS_TO")
        kg.add_relation("Paper:P2", "Method:GNN", "USES_METHOD")
        kg.add_relation("Paper:P2", "Concept:ProtFold", "STUDIES")
        st.session_state.kg = kg
    return st.session_state.kg

kg: InMemoryKnowledgeGraph = get_graph_store()
visualizer = GraphVisualizer2D(height="620px")

# 2. Top Header Telemetry
top_col1, top_col2, top_col3 = st.columns([3, 1, 1])
top_col1.title("🔬 Hidden Scientific Links Discovery Engine")
top_col2.metric("Knowledge Base Nodes", len(kg.graph.nodes))
top_col3.metric("Indexed Relations", len(kg.graph.edges))

# 3. Main Workspace Navigation Tabs
tab_discovery, tab_explorer, tab_ingestion = st.tabs([
    "🚀 Agentic Discovery", 
    "🌐 2D Knowledge Graph Explorer", 
    "📄 Local Paper Ingestion"
])

# ==========================================
# TAB 1: AGENTIC DISCOVERY WORKSPACE
# ==========================================
with tab_discovery:
    st.subheader("Autonomous Link Reasoning Pipeline")
    
    query_input = st.text_input(
        "Enter discovery inquiry:", 
        value="What hidden methodological connection exists between Computer Vision and Structural Biology?"
    )
    
    if st.button("Run Discovery Agent", type="primary"):
        initial_state = {
            "user_query": query_input,
            "kg_reference": kg,
            "grounded_concepts": [],
            "candidate_bridges": [],
            "validated_links": [],
            "final_hypothesis": ""
        }

        # Create a status container
        with st.status("Initializing Discovery Agent...", expanded=True) as status:
            final_text = ""
            all_candidate_bridges = []
            import uuid
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            for event in agent_app.stream(initial_state, config, stream_mode="updates"):
                for node_name, state_update in event.items():
                    # --- PHASE 1: GROUNDING ---
                    if node_name == "grounding":
                        status.update(label="✅ Phase 1: Grounding Complete. Searching topology...")
                        concepts = state_update.get('grounded_concepts', [])
                        st.write(f"**Grounded Concepts:** `{', '.join(concepts)}`")
                        
                    # --- PHASE 2: DISCOVERY ---
                    elif node_name == "discovery":
                        status.update(label="✅ Phase 2: Structural Discovery Complete. Validating metrics...")
                        bridges = state_update.get('candidate_bridges', [])
                        all_candidate_bridges.extend(bridges)
                        st.write(f"**Candidate Bridges Found:** {len(bridges)}")
                        for b in bridges:
                            if "type" in b and "name" in b:
                                st.caption(f"↳ {b['type']}: {b['name']}")
                            elif "shared_method" in b:
                                st.caption(f"↳ Method: {b['shared_method']}")
                            
                    # --- PHASE 3: VALIDATION ---
                    elif node_name == "validation":
                        status.update(label="✅ Phase 3: Validation Complete. Reviewing findings...")
                        links = state_update.get('validated_links', [])
                        st.write("Calculated Adamic-Adar scores and temporal trends.")
                        for l in links:
                            st.caption(f"↳ Bridge: {l.get('bridge', 'Unknown')}")
                            
                    # --- PHASE 3.5: SELF-CORRECTION ---
                    elif node_name == "self_correction":
                        status.update(label="✅ Phase 3.5: Peer Review Complete. Synthesizing hypothesis...")
                        filtered_links = state_update.get('validated_links', [])
                        st.write("Adversarial review filtered out trivial connections.")
                        st.write(f"**High-Confidence Bridges Retained:** {len(filtered_links)}")
                            
                    # --- PHASE 4: SYNTHESIS ---
                    elif node_name == "synthesis":
                        status.update(label="✅ Phase 4: Synthesis Complete!", state="complete")
                        final_text = state_update.get("final_hypothesis", "")
                        
                    # Catch the early exit event
                    elif node_name == "early_exit":
                        status.update(label="🛑 Discovery Halted: No connections found.", state="complete")
                        final_text = state_update.get("final_hypothesis", "")

        # Inject AI Discovered Bridges into the central graph state
        for bridge in all_candidate_bridges:
            if bridge.get("type") == "SEMANTIC_BRIDGE":
                node_a = bridge.get("node_a")
                node_b = bridge.get("node_b")
                if node_a and node_b and not kg.graph.has_edge(node_a, node_b):
                    kg.add_relation(
                        node_a, 
                        node_b, 
                        "AGENT_DISCOVERY", 
                        {"rationale": bridge.get("rationale", "Injected via Agentic Semantic Bridge")}
                    )
                    st.toast("🔥 New Semantic Bridge injected into Knowledge Graph!", icon="🚀")

        # Display the final hypothesis
        st.markdown("### 💡 Final Scientific Hypothesis")
        def stream_text_typewriter(text):
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.02)
                
        st.write_stream(stream_text_typewriter(final_text))

# ==========================================
# TAB 2: 2D GRAPH EXPLORER (Interactive Canvas)
# ==========================================
with tab_explorer:
    col_ctrl, col_canvas = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("Graph Filters")
        
        mode = st.radio("View Mode", ["Full Graph", "k-Hop Ego Subgraph", "Shortest Path"])
        active_subgraph = kg.graph
        center_highlight = None
        
        if mode == "k-Hop Ego Subgraph":
            node_choice = st.selectbox("Center Node", list(kg.graph.nodes))
            k_hops = st.slider("Radius (Hops)", 1, 3, 1)
            active_subgraph = kg.get_ego_subgraph(node_choice, radius=k_hops)
            center_highlight = node_choice
            
        elif mode == "Shortest Path":
            n1 = st.selectbox("Source Entity", list(kg.graph.nodes), index=0)
            n2 = st.selectbox("Target Entity", list(kg.graph.nodes), index=min(1, len(kg.graph.nodes)-1))
            path_nodes = kg.find_shortest_path(n1, n2)
            if path_nodes:
                active_subgraph = kg.graph.subgraph([p["node"] for p in path_nodes]).copy()
                st.success(f"Path distance: {len(path_nodes)-1} hops")
            else:
                st.warning("No path detected.")
                active_subgraph = nx.MultiDiGraph()
                
        # Link Prediction Tool
        st.divider()
        st.subheader("Adamic-Adar Link Scorer")
        u = st.selectbox("Node A", list(kg.graph.nodes), key="u_node")
        v = st.selectbox("Node B", list(kg.graph.nodes), key="v_node")
        if st.button("Score Unconnected Pair"):
            score = kg.calculate_adamic_adar(u, v)
            st.metric("Adamic-Adar Proximity", f"{score:.4f}")

    with col_canvas:
        st.subheader("Graph Canvas")
        temp_html_path = "temp_vis.html"
        visualizer.export_html(active_subgraph, output_path=temp_html_path, center_node=center_highlight)
        
        with open(temp_html_path, "r", encoding="utf-8") as f:
            raw_html = f.read()
            
        components.html(raw_html, height=640)
        
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)

# ==========================================
# TAB 3: LOCAL PAPER INGESTION
# ==========================================
with tab_ingestion:
    st.subheader("Local PDF Ingestion & Entity Extraction")
    
    uploaded_pdfs = st.file_uploader("Upload Scientific Papers (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Extract Entities to In-Memory Graph", type="primary") and uploaded_pdfs:
        for uploaded_pdf in uploaded_pdfs:
            with st.status(f"Processing {uploaded_pdf.name}...", expanded=True) as status:
                st.write("1. Saving and reading document layout...")
                
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getbuffer())
                    temp_pdf_path = tmp.name
                    
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(temp_pdf_path)
                    pages = loader.load()
                    full_text = "\n".join([page.page_content for page in pages])
                    st.write(f"✅ Extracted {len(pages)} pages ({len(full_text.split())} words).")
                    
                    st.write("2. Extracting Paper Metadata (Title, Year, Description)...")
                    from src.agent.nodes import get_llm
                    title_llm = get_llm()
                    import json
                    meta_prompt = f"Extract the exact title, publication year (as an integer, default to 2026 if absent), and a short 2-3 sentence abstract/description of the scientific paper from this text. Return your response strictly as a JSON object with keys 'title', 'year', 'description', with no markdown formatting.\n\n{pages[0].page_content[:2000]}"
                    meta_raw = title_llm.invoke(meta_prompt).content.strip()
                    if meta_raw.startswith("```json"): meta_raw = meta_raw[7:]
                    elif meta_raw.startswith("```"): meta_raw = meta_raw[3:]
                    if meta_raw.endswith("```"): meta_raw = meta_raw[:-3]
                    try:
                        meta_json = json.loads(meta_raw.strip())
                        paper_title = meta_json.get("title", "Unknown Title")
                        paper_year = meta_json.get("year", 2026)
                        paper_desc = meta_json.get("description", "")
                    except:
                        paper_title = meta_raw[:100] # Fallback
                        paper_year = 2026
                        paper_desc = "Description parsing failed."
                    st.write(f"✅ Metadata identified: **{paper_title}** ({paper_year})")
                    
                    st.write("3. Local Preprocessing (scispaCy Sentence Segmentation)...")
                
                    from src.tools.text_processing import extract_dense_sentences
                    chunk = extract_dense_sentences(full_text, max_chars=4000)
                    st.write("   - scispaCy filtered raw sentences down to dense sentences containing unique scientific entities.")
                
                    st.write("4. Structured LLM Triplet Extraction...")
                
                    from pydantic import BaseModel, Field
                    from typing import List
                    from rapidfuzz import fuzz
                    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
                
                    class ExtractedGraphTriplet(BaseModel):
                        subject: str = Field(description="The entity doing the action (usually the paper name)")
                        predicate: str = Field(description="Relationship type: USES_METHOD, STUDIES, BELONGS_TO, AUTHORED_BY, AFFILIATED_WITH")
                        object: str = Field(description="The target entity name, e.g., 'Graph Neural Networks', 'Jane Doe', or 'Stanford University'")
                        object_type: str = Field(description="The type of the object: Concept, Method, Field, Dataset, Author, or Organization")
                        section: str = Field(description="The section this was extracted from: Introduction, Methodology, or Related Work")

                    class DocumentExtractionSchema(BaseModel):
                        triplets: List[ExtractedGraphTriplet]
                
                    # Initialize LLM
                    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
                        repo_id="Qwen/Qwen2.5-72B-Instruct",
                        task="text-generation",
                        max_new_tokens=4096,
                        huggingfacehub_api_token=hf_token
                    ))
                
                    prompt = f"""
    Extract scientific relationships from this text chunk into structured triplets. 
    The paper is titled '{paper_title}'. Make sure to extract Author and Organization entities if present.

    Output your response strictly as a JSON object matching this schema, with no markdown formatting:
    {{
        "triplets": [
            {{
                "subject": "string (entity doing the action)",
                "predicate": "string (USES_METHOD, STUDIES, BELONGS_TO, AUTHORED_BY, AFFILIATED_WITH)",
                "object": "string (target entity)",
                "object_type": "string (Concept, Method, Field, Dataset, Author, or Organization)",
                "section": "string (Introduction, Methodology, or Related Work)"
            }}
        ]
    }}

    {chunk}
    """
                
                    st.write("   - Calling Qwen 2.5 API...")
                    raw_response = llm.invoke(prompt).content
                
                    try:
                        import json
                        raw_clean = raw_response.strip()
                        if raw_clean.startswith("```json"):
                            raw_clean = raw_clean[7:]
                        elif raw_clean.startswith("```"):
                            raw_clean = raw_clean[3:]
                        if raw_clean.endswith("```"):
                            raw_clean = raw_clean[:-3]
                        
                        parsed_json = json.loads(raw_clean.strip())
                        extraction = DocumentExtractionSchema(**parsed_json)
                    except Exception as e:
                        raise ValueError(f"Failed to parse LLM JSON: {e}\nRaw Output: {raw_response}")
                
                    st.write("5. Canonicalization and Graph Injection...")
                
                    # Build canonical map of existing entities
                    existing_entities = {n: d.get("name", n) for n, d in kg.graph.nodes(data=True)}
                
                    new_paper_id = f"Paper:{len(kg.graph.nodes)+1}"
                    kg.add_entity(new_paper_id, "Paper", {"name": paper_title, "year": paper_year, "description": paper_desc})
                
                    for trip in extraction.triplets:
                        # Fuzzy match the object
                        extracted_clean = trip.object.strip().lower()
                        canonical_id = None
                        
                        from rapidfuzz import process, fuzz, utils
                        match = process.extractOne(extracted_clean, list(existing_entities.values()), scorer=fuzz.ratio, processor=utils.default_process)
                        if match and match[1] >= 85:
                            for k, v in existing_entities.items():
                                if v == match[0]:
                                    canonical_id = k
                                    break
                            
                        if not canonical_id:
                            # Create new unique node
                            canonical_id = f"{trip.object_type}:{trip.object.replace(' ', '')}"
                            kg.add_entity(canonical_id, trip.object_type, {"name": trip.object})
                            existing_entities[canonical_id] = trip.object
                            st.write(f"   - Added New Node: {trip.object}")
                        else:
                            st.write(f"   - Mapped to Canonical Node: {existing_entities[canonical_id]}")
                        
                        # Weight modifier based on section heuristic
                        weight = 1.0
                        if trip.section.lower() == "methodology": weight = 1.5
                        elif trip.section.lower() == "related work": weight = 0.8
                    
                        kg.add_relation(new_paper_id, canonical_id, trip.predicate, {"weight": weight, "section": trip.section})

                    st.write("6. Cross-Paper Auto-Linking...")
                    from src.tools.grounding import get_cross_encoder
                    try:
                        encoder = get_cross_encoder()
                        existing_papers = []
                        for n, data in kg.graph.nodes(data=True):
                            if n.startswith("Paper:") and n != new_paper_id:
                                existing_papers.append((n, data.get("description", data.get("name", "")), data.get("name", n)))
                        
                        semantic_edges_added = 0
                        if paper_desc and existing_papers:
                            pairs = [[paper_desc, desc] for _, desc, _ in existing_papers]
                            scores = encoder.predict(pairs)
                            
                            for i, score in enumerate(scores):
                                if float(score) > 0.6:  # High similarity threshold for stsb-roberta-base
                                    target_node = existing_papers[i][0]
                                    target_desc = existing_papers[i][1]
                                    target_name = existing_papers[i][2]
                                    
                                    # Ask LLM for a meaningful relation name instead of 'SEMANTIC_SIMILAR'
                                    rel_prompt = f"These two scientific papers are highly related. \nPaper 1: {paper_title}\nAbstract: {paper_desc}\n\nPaper 2: {target_name}\nAbstract: {target_desc}\n\nWhat is the specific relationship between Paper 1 and Paper 2? Respond with ONLY a single UPPERCASE snake_case string (e.g. BUILDS_UPON, SHARES_METHODOLOGY, TACKLES_SAME_PROBLEM)."
                                    try:
                                        relation_name = title_llm.invoke(rel_prompt).content.strip().replace(' ', '_').upper()
                                        if not relation_name or len(relation_name) > 40:
                                            relation_name = "HIGHLY_RELATED_TO"
                                    except:
                                        relation_name = "HIGHLY_RELATED_TO"
                                        
                                    kg.add_relation(new_paper_id, target_node, relation_name, {"score": float(score), "rationale": "Ingestion auto-linking"})
                                    semantic_edges_added += 1
                        st.write(f"   - Added {semantic_edges_added} dynamic semantic edges to existing papers.")
                    except Exception as ce_e:
                        st.write(f"   - Auto-linking skipped or failed: {ce_e}")

                    status.update(label=f"Ingestion Complete for {uploaded_pdf.name}!", state="complete")
                except Exception as e:
                    status.update(label=f"Ingestion Failed for {uploaded_pdf.name}: {e}", state="error")
                finally:
                    if os.path.exists(temp_pdf_path):
                        try:
                            os.remove(temp_pdf_path)
                        except:
                            pass
                    
        st.rerun()
