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
        
        # In production/deployment, we start with a true empty state.
        # Only load sample data if explicitly requested via a demo toggle (omitted for clean startup).
        st.session_state.kg = kg
    return st.session_state.kg

kg: InMemoryKnowledgeGraph = get_graph_store()
visualizer = GraphVisualizer2D(height="620px")

# 2. Top Header
st.markdown("""
# Scientific Research Explorer
Explore papers, methods, datasets, citations, and research connections.
""")
st.divider()

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

# Count entity types
paper_count = sum(1 for _, d in kg.graph.nodes(data=True) if d.get('label') == 'Paper')
author_count = sum(1 for _, d in kg.graph.nodes(data=True) if d.get('label') == 'Author')

col_stat1.metric("Papers", f"{paper_count:,}")
col_stat2.metric("Entities", f"{len(kg.graph.nodes):,}")
col_stat3.metric("Relationships", f"{len(kg.graph.edges):,}")
col_stat4.metric("Researchers", f"{author_count:,}")

st.write("") # Spacer

# 3. Main Workspace Navigation Tabs
nav_selection = st.sidebar.radio("Navigation", ["Research Agent", "Knowledge Graph", "Local PDF Ingestion", "Graph Editor"])

# ==========================================
# TAB 1: RESEARCH ASSISTANT
# ==========================================
if nav_selection == "Research Agent":
    if len(kg.graph.nodes) == 0:
        st.info("Ask your first research question once you add papers.")
        st.caption('Try: "Which methods are most commonly used for traffic prediction?"')
    
    st.markdown("### Ask a research question")
    query_input = st.text_input(
        "Question", 
        label_visibility="collapsed",
        placeholder="How are GNNs related to traffic prediction?",
        value="What hidden methodological connection exists between Computer Vision and Structural Biology?" if len(kg.graph.nodes) > 0 else ""
    )
    
    if st.button("Analyze", type="primary") and query_input:
        initial_state = {
            "user_query": query_input,
            "kg_reference": kg,
            "grounded_concepts": [],
            "candidate_bridges": [],
            "validated_links": [],
            "final_hypothesis": ""
        }

        with st.status("Analyzing...", expanded=True) as status:
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
                    import datetime
                    agent_prov = {
                        "document_id": None,
                        "extraction_method": "agent_synthesis",
                        "created_at": datetime.datetime.now().isoformat(),
                        "created_by": None
                    }
                    kg.add_relation(
                        node_a, 
                        node_b, 
                        "AGENT_DISCOVERY", 
                        source="algorithm",
                        confidence=bridge.get("similarity_score", 0.5),
                        provenance=agent_prov,
                        properties={"rationale": bridge.get("rationale", "Injected via Agentic Semantic Bridge")}
                    )
                    st.toast("🔥 New Semantic Bridge injected into Knowledge Graph!", icon="🚀")

        # Display the structured answer
        st.markdown("### Answer")
        st.markdown("────────────────────────────────────────")
        
        # We can format the final text to replace any markdown tags with HTML chips if needed
        # but for now we rely on the agent formatting.
        def stream_text_typewriter(text):
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.02)
                
        st.write_stream(stream_text_typewriter(final_text))

# ==========================================
# TAB 2: KNOWLEDGE GRAPH EXPLORER
# ==========================================
elif nav_selection == "Knowledge Graph":
    if len(kg.graph.nodes) == 0:
        st.info("Your graph is empty. Add papers to begin discovering connections.")
    else:
        # SIDEBAR: Graph Filters
        st.sidebar.divider()
        st.sidebar.markdown("### Graph Filters")
        mode = st.sidebar.radio("View Mode", ["Full Graph", "Neighborhood", "Shortest Path"])
        
        active_subgraph = kg.graph
        center_highlight = None
        
        if mode == "Neighborhood":
            node_choice = st.sidebar.selectbox("Center Node", list(kg.graph.nodes))
            k_hops = st.sidebar.slider("Connection depth", 1, 3, 1)
            active_subgraph = kg.get_ego_subgraph(node_choice, radius=k_hops)
            center_highlight = node_choice
            
        elif mode == "Shortest Path":
            n1 = st.sidebar.selectbox("Source Entity", list(kg.graph.nodes), index=0)
            n2 = st.sidebar.selectbox("Target Entity", list(kg.graph.nodes), index=min(1, len(kg.graph.nodes)-1))
            path_nodes = kg.find_shortest_path(n1, n2)
            if path_nodes:
                active_subgraph = kg.graph.subgraph([p["node"] for p in path_nodes]).copy()
                st.sidebar.success(f"Path distance: {len(path_nodes)-1} hops")
            else:
                st.sidebar.warning("No path detected.")
                active_subgraph = nx.MultiDiGraph()
                
        # SIDEBAR: Graph Analytics
        st.sidebar.divider()
        st.sidebar.markdown("### Graph Analytics")
        st.sidebar.markdown("**Find connections between**")
        u = st.sidebar.selectbox("First entity", list(kg.graph.nodes), key="u_node")
        v = st.sidebar.selectbox("Second entity", list(kg.graph.nodes), key="v_node")
        
        if st.sidebar.button("Analyze Connection", type="primary"):
            score = kg.calculate_adamic_adar(u, v)
            st.sidebar.markdown(f"**Confidence:** {score*100:.1f}%")
            st.sidebar.caption("Scoring method: Adamic-Adar")

        # MAIN CONTENT AREA: Graph Canvas
        st.markdown("**Legend**")
        legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px;">'
        for label, colors in COLOR_PALETTE.items():
            if label == "Default":
                continue
            legend_html += f'<span style="display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: #94A3B8;"><span style="width: 10px; height: 10px; border-radius: 50%; background: {colors["background"]}; display: inline-block;"></span>{label}</span>'
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)
    
        temp_html_path = "temp_vis.html"
        visualizer.export_html(active_subgraph, output_path=temp_html_path, center_node=center_highlight)
        
        with open(temp_html_path, "r", encoding="utf-8") as f:
            raw_html = f.read()
            
        import streamlit.components.v1 as components
        components.html(raw_html, height=640, scrolling=False)
        
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)
            
        # MAIN CONTENT AREA: Temporal Chart
        st.markdown("### Papers published over time")
        
        # Aggregate unique papers per year
        from collections import defaultdict
        year_counts = defaultdict(int)
        seen_papers = set()
        
        for n, data in kg.graph.nodes(data=True):
            if data.get("label") == "Paper" and n not in seen_papers:
                seen_papers.add(n)
                if "year" in data and data["year"] is not None:
                    try:
                        year = int(data["year"])
                        year_counts[year] += 1
                    except ValueError:
                        pass
                    
        if year_counts:
            sorted_trend = {str(k): year_counts[k] for k in sorted(year_counts)}
            render_temporal_chart(sorted_trend, "Papers", "All Fields")
        else:
            st.caption("No temporal data available yet.")

# ==========================================
# TAB 3: LOCAL PAPER INGESTION
# ==========================================
elif nav_selection == "Local PDF Ingestion":
    st.markdown("### Add Research Papers")
    st.caption("Upload one or more scientific PDF files to extract entities and build the knowledge graph.")
    
    uploaded_pdfs = st.file_uploader("Drop PDF files here or click to browse", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("Start Processing", type="primary") and uploaded_pdfs:
        for uploaded_pdf in uploaded_pdfs:
            # We don't know the parser yet, so we just start the spinner
            with st.status(f"Analyzing paper: {uploaded_pdf.name}", expanded=True) as status:
                st.write("✓ Reading document")
                
                import tempfile
                import shutil
                
                # Persist the file for Graph Editor (FR-25)
                upload_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                
                safe_filename = uploaded_pdf.name.replace(" ", "_")
                temp_pdf_path = os.path.join(upload_dir, safe_filename)
                
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                    
                try:
                    # 1. & 2. Smart Router Pipeline
                    from src.ingestion.router import route_and_parse
                    from src.ingestion.pymupdf_parser import PyMuPDFParser
                    parse_result = route_and_parse(temp_pdf_path, filename=uploaded_pdf.name)
                    
                    if parse_result.parser != "pymupdf":
                        st.write("✓ Detecting structure")
                    st.write("● Extracting structured document metadata")
                    
                    # 3. Pure Structured Document Extraction (No LLM / No spaCy)
                    fast_parser = PyMuPDFParser()
                    paper_schema = fast_parser.extract_paper(temp_pdf_path)
                    
                    # If Docling was used, override with any enhanced text if available
                    if parse_result.parser in ("docling", "docling_ocr") and parse_result.text:
                        docling_title = fast_parser.extract_title(parse_result.text)
                        if docling_title and docling_title != "Untitled Paper":
                            paper_schema.title = docling_title
                        docling_abstract = fast_parser.extract_abstract(parse_result.text)
                        if docling_abstract:
                            paper_schema.abstract = docling_abstract
                        docling_cites = fast_parser.extract_references(parse_result.text)
                        if docling_cites:
                            paper_schema.cites = docling_cites

                    # Document Sections
                    doc_sections = fast_parser.extract_sections(parse_result.text)
                    
                    paper_title = paper_schema.title
                    paper_desc = paper_schema.abstract
                    is_partial = False
                    
                    st.write("○ Updating knowledge graph")
                    
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
                        
                    import datetime
                    
                    prov_node = {"provenance": {"document": uploaded_pdf.name}}

                    paper_doi_norm = normalize_doi(paper_schema.doi)
                    paper_title_norm = normalize_title(paper_schema.title)
                    
                    # Create the core paper node
                    new_paper_id = f"Paper:{paper_doi_norm if paper_doi_norm else paper_title_norm + str(paper_schema.year)}"
                    new_paper_id = new_paper_id.replace(' ', '')
                    
                    kg.add_entity(new_paper_id, "Paper", {
                        "name": paper_title, 
                        "title_norm": paper_title_norm,
                        "year": paper_schema.year, 
                        "description": paper_desc,
                        "doi": paper_doi_norm,
                        "partial_ingestion": is_partial,
                        "file_path": temp_pdf_path,
                        **prov_node
                    })
                    
                    # Add authors (from structured document metadata)
                    for author in paper_schema.authors:
                        author_norm = normalize_name(author.name)
                        if author_norm:
                            author_id = f"Author:{author_norm.replace(' ', '')}"
                            kg.add_entity(author_id, "Author", {"name": author.name, "name_norm": author_norm, **prov_node})
                            auth_prov = {
                                "document_id": uploaded_pdf.name,
                                "extraction_method": "document_metadata",
                                "created_at": datetime.datetime.now().isoformat(),
                                "created_by": None
                            }
                            kg.add_relation(author_id, new_paper_id, "AUTHORED", source="extracted", confidence=1.0, provenance=auth_prov)

                    # Add document sections (HAS_SECTION)
                    for sec_name in doc_sections:
                        sec_norm = normalize_name(sec_name)
                        if sec_norm:
                            sec_id = f"Section:{new_paper_id}_{sec_norm.replace(' ', '')[:30]}"
                            kg.add_entity(sec_id, "Section", {"name": sec_name, "paper_id": new_paper_id, **prov_node})
                            sec_prov = {
                                "document_id": uploaded_pdf.name,
                                "extraction_method": "document_structure",
                                "created_at": datetime.datetime.now().isoformat(),
                                "created_by": None
                            }
                            kg.add_relation(new_paper_id, sec_id, "HAS_SECTION", source="extracted", confidence=1.0, provenance=sec_prov)

                    # Paper citation relationships
                    for citation in paper_schema.cites:
                        cite_norm = normalize_title(citation)
                        if len(cite_norm) > 5:
                            cite_id = None
                            from rapidfuzz import process, fuzz
                            existing_papers = [(n, d.get("title_norm", normalize_title(d.get("name", "")))) for n, d in kg.graph.nodes(data=True) if n.startswith("Paper:")]
                            if existing_papers:
                                match = process.extractOne(cite_norm, [p[1] for p in existing_papers], scorer=fuzz.ratio)
                                if match and match[1] >= 90:
                                    cite_id = existing_papers[match[2]][0]
                            if not cite_id:
                                cite_id = f"Paper:{cite_norm.replace(' ', '')[:40]}"
                                kg.add_entity(cite_id, "Paper", {"name": citation, "title_norm": cite_norm, "stub": True, **prov_node})
                            cite_prov = {
                                "document_id": uploaded_pdf.name,
                                "extraction_method": "bibliography_parser",
                                "created_at": datetime.datetime.now().isoformat(),
                                "created_by": None
                            }
                            kg.add_relation(new_paper_id, cite_id, "CITES", source="extracted", confidence=0.95, provenance=cite_prov)
                        
                    st.write(f"✅ Successfully ingested into Knowledge Graph: **{paper_schema.title}**")

                    # 4. Eager Cache Warming (NFR-04)
                    st.write("4. Warming Vector Cache...")
                    try:
                        new_node_ids = [new_paper_id]
                        new_node_ids.extend([f"Author:{normalize_name(a.name).replace(' ', '')}" for a in paper_schema.authors if normalize_name(a.name)])
                        kg.vectors.get_embeddings(list(set(new_node_ids)))
                    except Exception as e:
                        print(f"Warning: Cache warming failed: {e}")
                        
                    st.write("○ Discovering connections")
                    semantic_edges_added = 0
                    try:
                        from src.tools.grounding import get_cross_encoder
                        encoder = get_cross_encoder()
                        existing_papers = []
                        for n, data in kg.graph.nodes(data=True):
                            if n.startswith("Paper:") and not data.get("stub") and n != new_paper_id:
                                existing_papers.append((n, data.get("description", data.get("name", "")), data.get("name", n)))
                        
                        if paper_desc and existing_papers:
                            pairs = [[paper_desc, desc] for _, desc, _ in existing_papers]
                            scores = encoder.predict(pairs)
                            
                            for i, score in enumerate(scores):
                                if float(score) > 0.6:  # High similarity threshold
                                    target_node = existing_papers[i][0]
                                    sem_prov = {
                                        "document_id": None,
                                        "extraction_method": "cross_encoder_semantic",
                                        "created_at": datetime.datetime.now().isoformat(),
                                        "created_by": None
                                    }
                                    kg.add_relation(
                                        new_paper_id, target_node, "SEMANTICALLY_SIMILAR",
                                        source="algorithm",
                                        confidence=max(0.0, min(1.0, float(score))),
                                        provenance=sem_prov,
                                        properties={"score": float(score)}
                                    )
                                    semantic_edges_added += 1
                        st.write(f"✓ Discovered {semantic_edges_added} semantic connection(s)")
                    except Exception as sim_err:
                        st.write("○ Connection discovery skipped")

                    if is_partial:
                        status.update(label=f"⚠️ Partial Ingestion: {uploaded_pdf.name}", state="complete")
                    else:
                        status.update(label=f"Processing Complete: {uploaded_pdf.name}", state="complete")
                    
                    # ---------------------------------------------------------
                    # POST PROCESSING SUMMARY
                    # ---------------------------------------------------------
                    triage_meta = parse_result.metadata.get("triage", {})
                    reason = "Text-rich layout" if parse_result.parser == "pymupdf" else ("Scanned document" if parse_result.parser == "docling_ocr" else "Complex scientific layout")
                    total_time = sum(parse_result.timings.values()) / 1000
                    
                    st.markdown("### Processing complete" if not is_partial else "### Partial Ingestion")
                    st.markdown(f"**{triage_meta.get('page_count', 0)} pages** · **{total_time:.2f} s**")
                    
                    st.markdown(f"""
                    <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 16px; margin: 10px 0;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 13px;">
                            <div><span style="color: #64748B;">Processing method<br></span> <b style="color: #E2E8F0;">{parse_result.parser.upper()}</b></div>
                            <div><span style="color: #64748B;">Information extraction<br></span> <b style="color: #E2E8F0;">Structured Parsing</b></div>
                            <div><span style="color: #64748B;">Extraction service<br></span> <b style="color: #10B981;">Local / Offline</b></div>
                            <div><span style="color: #64748B;">Fallback<br></span> <b style="color: #E2E8F0;">None</b></div>
                            <div><span style="color: #64748B;">Extraction quality<br></span> <b style="color: #E2E8F0;">{parse_result.quality_score * 100:.0f}%</b></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Extraction Summary
                    citation_count = len(paper_schema.cites)
                    section_count = len(doc_sections)
                    st.markdown("#### Extracted information")
                    st.markdown(f"""
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px;">
                        <div><b>1</b> Paper</div>
                        <div><b>{len(paper_schema.authors)}</b> Researchers</div>
                        <div><b>{section_count}</b> Document Sections</div>
                        <div><b>{citation_count}</b> Citations</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("▸ Processing details"):
                        st.markdown(f"""
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; font-family: monospace;">
                            <div>Triage</div> <div style="text-align: right;">{parse_result.timings.get('triage_ms', 0):.0f} ms</div>
                            <div>Parser</div> <div style="text-align: right;">{parse_result.timings.get('parser_ms', 0):.0f} ms</div>
                            <div>Quality gate</div> <div style="text-align: right;">{parse_result.timings.get('quality_gate_ms', 0):.0f} ms</div>
                            <div>Total</div> <div style="text-align: right;">{parse_result.timings.get('total_ms', 0):.0f} ms</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    status.update(label=f"Processing Failed for {uploaded_pdf.name}: {e}", state="error")
                finally:
                    pass
                    
        st.rerun()

# ==========================================
# TAB 4: GRAPH EDITOR
# ==========================================
elif nav_selection == "Graph Editor":
    st.markdown("### Interactive Graph Editor")
    
    # 1. Selection & Addition
    st.sidebar.markdown("### Editor Controls")
    editor_mode = st.sidebar.radio("Action", ["Edit Existing Node", "Add New Node"])
    
    if editor_mode == "Add New Node":
        st.markdown("#### Create New Entity")
        new_label = st.selectbox("Entity Type", ["Paper", "Author", "Method", "Dataset", "Organization", "Concept"])
        new_name = st.text_input("Name (Required)")
        new_desc = st.text_area("Description")
        
        if st.button("Create Node", type="primary"):
            if new_name.strip():
                # Use identical normalization to ingestion
                new_id = f"{new_label}:{new_name.strip()}"
                props = {"name": new_name.strip()}
                if new_desc.strip():
                    props["description"] = new_desc.strip()
                kg.add_entity(new_id, label=new_label, properties=props)
                st.success(f"Created node `{new_id}`")
                st.rerun()
            else:
                st.error("Name is required.")
                
    else:
        # Edit Existing Node
        node_options = list(kg.graph.nodes)
        if not node_options:
            st.info("The graph is empty.")
        else:
            selected_node = st.selectbox("Select Node to Edit", node_options)
            
            # Reset pending delete if selection changes
            if "pending_delete_id" in st.session_state and st.session_state.pending_delete_id != selected_node:
                st.session_state.pending_delete_id = None
                
            node_data = dict(kg.graph.nodes[selected_node])
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"#### Edit `{selected_node}`")
                
                # File Reading (FR-25)
                if node_data.get("label") == "Paper" and "file_path" in node_data:
                    st.info(f"Associated File: {os.path.basename(node_data['file_path'])}")
                    # For local rendering, we'd need to serve it or provide a download link
                    # Here we just show the sanitized path.
                    
                # Web Search (FR-28)
                search_query = node_data.get("name", selected_node).replace(" ", "+")
                st.link_button("Search Web (Google Scholar)", f"https://scholar.google.com/scholar?q={search_query}")
                
                with st.form("edit_node_form"):
                    updated_props = {}
                    for k, v in node_data.items():
                        if k == "label":
                            continue # Don't edit labels here to prevent breaking ID semantics
                        if isinstance(v, str):
                            if len(v) > 50:
                                updated_props[k] = st.text_area(f"{k.capitalize()}", value=v)
                            else:
                                updated_props[k] = st.text_input(f"{k.capitalize()}", value=v)
                        elif isinstance(v, (int, float)):
                            updated_props[k] = st.text_input(f"{k.capitalize()} (numeric)", value=str(v))
                        else:
                            st.write(f"*{k}:* `{v}` (Complex type, read-only)")
                            
                    if st.form_submit_button("Save Changes"):
                        kg.edit_entity(selected_node, updated_props)
                        st.success("Changes saved.")
                        st.rerun()
                        
            with col2:
                st.markdown("#### Danger Zone")
                # Two-step UX logic
                is_pending = st.session_state.get("pending_delete_id") == selected_node
                
                if not is_pending:
                    if st.button("Delete Node", type="secondary"):
                        st.session_state.pending_delete_id = selected_node
                        st.rerun()
                else:
                    st.warning("Are you sure? This deletes the node and all connected edges.")
                    col2a, col2b = st.columns(2)
                    with col2a:
                        if st.button("Confirm Delete", type="primary"):
                            kg.delete_entity(selected_node)
                            st.session_state.pending_delete_id = None
                            st.rerun()
                    with col2b:
                        if st.button("Cancel"):
                            st.session_state.pending_delete_id = None
                            st.rerun()
                            
            st.divider()
            
            # Edge Authoring (FR-24, FR-26)
            st.markdown("#### Relationships")
            edges = list(kg.graph.edges(selected_node, data=True))
            if edges:
                for src, dst, data in edges:
                    rel = data.get("relation", "UNKNOWN")
                    col_rel1, col_rel2 = st.columns([3, 1])
                    with col_rel1:
                        st.write(f"→ `{dst}` **({rel})**")
                    with col_rel2:
                        if st.button("Remove", key=f"del_edge_{src}_{dst}_{rel}"):
                            kg.delete_relation(src, dst, rel)
                            st.rerun()
            else:
                st.write("No edges connected to this node.")
                
            st.markdown("##### Add New Relationship")
            with st.form("add_edge_form"):
                target_node = st.selectbox("Target Node", [n for n in node_options if n != selected_node])
                rel_type = st.text_input("Relation Type (e.g., CITES, USES_METHOD)", value="CITES")
                
                if st.form_submit_button("Add Edge"):
                    if rel_type.strip() and target_node:
                        import datetime
                        user_prov = {
                            "document_id": None,
                            "extraction_method": "manual_edit",
                            "created_at": datetime.datetime.now().isoformat(),
                            "created_by": "streamlit_user"
                        }
                        kg.add_relation(
                            selected_node, target_node, rel_type.strip(),
                            source="user",
                            confidence=1.0,
                            provenance=user_prov
                        )
                        st.success("Edge added!")
                        st.rerun()
