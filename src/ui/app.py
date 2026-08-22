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
                    from src.agent.nodes import get_llm
                    title_llm = get_llm()
                    
                    from src.ingestion.docling_parser import DoclingIngestor
                    ingestor = DoclingIngestor()
                    
                    st.write("2. Parsing document structure using Docling...")
                    doc = ingestor.parse_pdf(temp_pdf_path)
                    
                    st.write("3. Extracting knowledge graph via LLM (Qwen 72B)...")
                    paper_schema = ingestor.extract_graph(doc)
                    paper_title = paper_schema.title
                    paper_desc = paper_schema.abstract
                    
                    st.write("3b. Extracting metadata entities via spaCy NER...")
                    from src.tools.text_processing import extract_dense_sentences_and_entities, normalize_date_to_year
                    full_text = doc.export_to_markdown()
                    _, metadata_entities = extract_dense_sentences_and_entities(full_text, max_chars=6000)
                    
                    st.write("4. Canonicalization, Normalization, and Graph Injection...")
                    
                    # Normalization helpers
                    import re
                    def normalize_doi(doi: str) -> str:
                        if not doi: return ""
                        d = doi.lower().strip()
                        d = re.sub(r'^https?://doi\.org/', '', d)
                        d = re.sub(r'^doi:', '', d)
                        return d.strip()

                    def normalize_title(title: str) -> str:
                        if not title: return ""
                        t = title.lower().strip()
                        t = re.sub(r'\s+', ' ', t)
                        t = re.sub(r'[^\w\s]', '', t)
                        return t

                    def normalize_name(name: str) -> str:
                        if not name: return ""
                        return re.sub(r'\s+', ' ', name.lower().strip())
                        
                    prov_node = {"provenance": {"document": uploaded_pdf.name}}
                    prov_edge = {"source": "extracted", "confidence": 1.0, "provenance": {"document": uploaded_pdf.name}}
                    
                    # Try to extract a real publication year from spaCy DATE entities
                    extracted_year = paper_schema.year
                    for ent in metadata_entities:
                        if ent["label"] == "DATE":
                            parsed_year = normalize_date_to_year(ent["text"])
                            if parsed_year and 1900 <= parsed_year <= 2030:
                                extracted_year = parsed_year
                                break  # Use the first plausible year
                    
                    paper_doi_norm = normalize_doi(paper_schema.doi)
                    paper_title_norm = normalize_title(paper_schema.title)
                    
                    # Create the core paper node
                    new_paper_id = f"Paper:{paper_doi_norm if paper_doi_norm else paper_title_norm + str(extracted_year)}"
                    new_paper_id = new_paper_id.replace(' ', '')
                    
                    kg.add_entity(new_paper_id, "Paper", {
                        "name": paper_title, 
                        "title_norm": paper_title_norm,
                        "year": extracted_year, 
                        "description": paper_desc,
                        "doi": paper_doi_norm,
                        **prov_node
                    })
                    
                    # Add authors from LLM extraction
                    seen_authors = set()
                    for author in paper_schema.authors:
                        author_norm = normalize_name(author.name)
                        author_id = f"Author:{author_norm.replace(' ', '')}"
                        kg.add_entity(author_id, "Author", {"name": author.name, "name_norm": author_norm, **prov_node})
                        kg.add_relation(author_id, new_paper_id, "AUTHORED", prov_edge)
                        seen_authors.add(author_norm)
                    
                    # Merge spaCy PERSON entities as additional authors (deduplicated)
                    spacy_prov = {"source": "extracted", "confidence": 0.85, "provenance": {"document": uploaded_pdf.name, "extractor": "spacy_ner"}}
                    for ent in metadata_entities:
                        if ent["label"] == "PERSON":
                            name_norm = normalize_name(ent["text"])
                            if name_norm not in seen_authors and len(name_norm) > 3:
                                author_id = f"Author:{name_norm.replace(' ', '')}"
                                kg.add_entity(author_id, "Author", {"name": ent["text"], "name_norm": name_norm, **prov_node})
                                kg.add_relation(author_id, new_paper_id, "AUTHORED", spacy_prov)
                                seen_authors.add(name_norm)
                    
                    # Add organizations from spaCy ORG entities
                    for ent in metadata_entities:
                        if ent["label"] == "ORG":
                            org_norm = normalize_name(ent["text"])
                            if len(org_norm) > 2:
                                org_id = f"Organization:{org_norm.replace(' ', '')}"
                                kg.add_entity(org_id, "Organization", {"name": ent["text"], **prov_node})
                                kg.add_relation(org_id, new_paper_id, "AFFILIATED_WITH", spacy_prov)
                        
                    # Add methods
                    for method in paper_schema.uses_methods:
                        method_id = f"Method:{method.name.replace(' ', '')}"
                        kg.add_entity(method_id, "Method", {"name": method.name, "category": method.category, **prov_node})
                        kg.add_relation(new_paper_id, method_id, "USES_METHOD", prov_edge)
                        
                    # Add datasets
                    for dataset in paper_schema.uses_datasets:
                        dataset_id = f"Dataset:{dataset.name.replace(' ', '')}"
                        kg.add_entity(dataset_id, "Dataset", {"name": dataset.name, **prov_node})
                        kg.add_relation(new_paper_id, dataset_id, "USES_DATASET", prov_edge)
                        
                    # Add problems
                    for problem in paper_schema.addresses_problems:
                        problem_id = f"ResearchProblem:{problem.name.replace(' ', '')}"
                        kg.add_entity(problem_id, "ResearchProblem", {"name": problem.name, **prov_node})
                        kg.add_relation(new_paper_id, problem_id, "ADDRESSES", prov_edge)
                        
                    # Add metrics
                    for metric in paper_schema.evaluated_by:
                        metric_id = f"Metric:{metric.name.replace(' ', '')}"
                        kg.add_entity(metric_id, "Metric", {"name": metric.name, **prov_node})
                        kg.add_relation(new_paper_id, metric_id, "EVALUATED_BY", {"value": metric.value, **prov_edge})
                        
                    # Add results
                    for result in paper_schema.reports_results:
                        res_id = f"Result:{result.description.replace(' ', '')[:40]}"
                        kg.add_entity(res_id, "Result", {"description": result.description, "improvement": result.improvement, **prov_node})
                        kg.add_relation(new_paper_id, res_id, "REPORTS", prov_edge)
                        
                    # Paper relationships helper
                    def link_papers(citations, rel_type):
                        for citation in citations:
                            cite_norm = normalize_title(citation)
                            cite_id = None
                            from rapidfuzz import process, fuzz
                            existing_papers = [(n, d.get("title_norm", normalize_title(d.get("name", "")))) for n, d in kg.graph.nodes(data=True) if n.startswith("Paper:")]
                            if existing_papers:
                                match = process.extractOne(cite_norm, [p[1] for p in existing_papers], scorer=fuzz.ratio)
                                if match and match[1] >= 90:
                                    cite_id = existing_papers[match[2]][0]
                            if not cite_id:
                                cite_id = f"Paper:{cite_norm.replace(' ', '')}"
                                kg.add_entity(cite_id, "Paper", {"name": citation, "title_norm": cite_norm, "stub": True, **prov_node})
                            kg.add_relation(new_paper_id, cite_id, rel_type, prov_edge)

                    link_papers(paper_schema.cites, "CITES")
                    link_papers(paper_schema.builds_on, "BUILDS_ON")
                    link_papers(paper_schema.extends, "EXTENDS")
                    link_papers(paper_schema.compares_to, "COMPARES")
                    link_papers(paper_schema.contradicts, "CONTRADICTS")
                        
                    st.write(f"✅ Successfully ingested into NetworkX: **{paper_schema.title}**")
                    st.write(f"   - Authors (LLM + spaCy): {len(seen_authors)} | Orgs (spaCy): {sum(1 for e in metadata_entities if e['label'] == 'ORG')} | Year: {extracted_year}")

                    st.write("6. Cross-Paper Auto-Linking...")
                    from src.tools.grounding import get_cross_encoder
                    try:
                        encoder = get_cross_encoder()
                        existing_papers = []
                        for n, data in kg.graph.nodes(data=True):
                            if n.startswith("Paper:") and not data.get("stub") and n != new_paper_id:
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
                                        
                                    kg.add_relation(new_paper_id, target_node, relation_name, {"score": float(score), "source": "algorithm", "rationale": "Ingestion auto-linking"})
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
