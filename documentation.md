# Hidden Scientific Links Discovery Engine (HSLDE) Documentation

## 1. Introduction
The **Hidden Scientific Links Discovery Engine (HSLDE)** is an advanced, AI-powered interactive platform designed to discover, reason about, and visualize implicit connections and methodological bridges between disparate scientific domains. The project leverages Large Language Models (LLMs), natural language processing (NLP), knowledge graphs (KG), and multi-agent workflows to synthesize novel scientific hypotheses from existing literature.

---

## 2. Functional Requirements
The system must fulfill the following functional capabilities:
- **Interactive Knowledge Graph Management:** Maintain an in-memory knowledge graph that stores scientific entities (Papers, Methods, Concepts, Fields) and the directed relationships between them (e.g., `USES_METHOD`, `STUDIES`, `BELONGS_TO`).
- **Agentic Autonomous Discovery:** Allow users to submit complex scientific inquiries. An autonomous pipeline must execute the following phases:
  - **Grounding Phase:** Expand user queries using LLMs and perform hybrid searches to locate exact concept matches within the graph database.
  - **Structural Discovery Phase:** Identify topological graph bridges such as shared methods, structural holes, or boundary authors. Fall back to bipartite semantic matching if topological bridges are absent.
  - **Validation Phase:** Calculate mathematical link probabilities (e.g., Adamic-Adar score) and analyze temporal citation trends to validate candidate bridges.
  - **Self-Correction Phase:** Perform an adversarial LLM-based peer review to eliminate false positives or trivial connections.
  - **Synthesis Phase:** Generate a final, rigorously reasoned scientific hypothesis.
- **2D Graph Exploration:** Provide an interactive canvas allowing users to visualize the full graph, specific $k$-hop ego subgraphs, and shortest paths between entities. It must also incorporate an ad-hoc link prediction tool.
- **Local Document Ingestion & Extraction:** Allow users to upload scientific papers in PDF format. The system must extract the title, segment and filter dense scientific sentences, leverage structured LLM outputs to extract Subject-Predicate-Object triplets, and inject them into the graph using fuzzy-matched canonicalization.

---

## 3. Non-Functional Requirements
- **Performance & Context Management:** PDF ingestion and NLP routines must compress document texts efficiently (e.g., maximum 4000-character chunks) to avoid blowing out LLM context windows. In-memory operations must compute rapidly to ensure immediate UI feedback.
- **Scalability & Responsiveness:** The agentic workflow must stream updates progressively to the UI, ensuring the user remains informed about long-running reasoning processes.
- **Accuracy & Fallbacks:** Entity matching must employ robust fuzzy-matching algorithms (minimum 85% confidence ratio) and fall back to semantic vector similarity when exact lexical matches fail.
- **Modularity:** The AI reasoning pipeline must be decoupled from the user interface, managed via a finite state machine architecture to allow interruptibility and memory persistence.
- **Aesthetics & Usability:** The dashboard must be visually striking, featuring custom themes, real-time telemetry headers, and interactive UI components (expanders, status containers, metric displays).

---

## 4. Interface Requirements
The user interface is a web-based dashboard built using Streamlit, comprising the following elements:
- **Global Header/Telemetry:** Displays the application title alongside live metrics detailing the total count of Knowledge Base Nodes and Indexed Relations.
- **Agentic Discovery Workspace (Tab 1):**
  - Text input for inquiry submission.
  - Execution button ("Run Discovery Agent").
  - Expandable status logs detailing the execution of the 4-phase LangGraph pipeline in real-time.
  - A typewriter-effect text stream for the final synthesized scientific hypothesis.
- **2D Knowledge Graph Explorer (Tab 2):**
  - **Control Sidebar:** Radio buttons to toggle between "Full Graph", "k-Hop Ego Subgraph" (with radius slider), and "Shortest Path" modes. Select boxes for source and target entity selection. A dedicated Adamic-Adar Link Scorer input form.
  - **Canvas Area:** A dynamic, interactive PyVis HTML rendering of the selected graph subgraph, capable of panning, zooming, and node highlighting.
- **Local Paper Ingestion (Tab 3):**
  - Drag-and-drop file uploader supporting single or multiple PDF files.
  - "Extract Entities" execution button.
  - Progressive log window showing ingestion stages: PDF parsing, title extraction, scispaCy filtering, Qwen 2.5 API triplets generation, and graph canonicalization.

---

## 5. Python Implementation Details
The application is structured into a modular Python codebase:

```text
sientificPapersAnalysis/
├── main.py                     # Application entry point to run Streamlit
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables (API keys)
└── src/
    ├── ui/                     # Frontend user interface
    │   ├── app.py              # Main Streamlit dashboard layout and state
    │   ├── components.py       # Reusable UI components (stepper, charts)
    │   └── styles.py           # Custom CSS styling
    ├── graph/                  # Knowledge Graph management
    │   ├── in_memory_store.py  # NetworkX wrapper for the graph database
    │   ├── visualizer.py       # PyVis HTML graph rendering
    │   ├── mock_data.py        # Seed data for initial graph
    │   └── db.py               # Database connections
    ├── agent/                  # Autonomous LLM workflow
    │   ├── workflow.py         # LangGraph state machine definition
    │   ├── nodes.py            # Logic for each workflow phase
    │   └── state.py            # TypedDict defining the agent's memory
    └── tools/                  # Analytical algorithms and capabilities
        ├── grounding.py        # Embeddings and BM25 RRF hybrid search
        ├── discovery.py        # Topological and semantic bridge discovery
        ├── validation.py       # Link probability and temporal analysis
        └── document_loader.py  # Local PDF parsing and entity extraction
```

### Key Components
- `main.py`: The application entry point which programmatically bootstraps the Streamlit server.
- `src/ui/app.py`: The frontend logic. It manages Streamlit state, renders the multi-tab layout, and coordinates event streams from the LangGraph workflow.
- `src/graph/in_memory_store.py`: Implements `InMemoryKnowledgeGraph` acting as a wrapper over `networkx.MultiDiGraph`. Handles node/edge insertion, subgraph extraction, and shortest-path queries.
- `src/graph/visualizer.py`: Facilitates the conversion of NetworkX graphs into interactive `pyvis` HTML canvases.
- `src/agent/workflow.py`: Defines the LangGraph `StateGraph`, wiring together nodes (functions) and conditional routing edges (e.g., early exits if no topological bridges are discovered).
- `src/agent/nodes.py`: Implements the discrete phases of the agentic pipeline. It binds LangChain tools to the `ChatHuggingFace` LLM, allowing the model to trigger database searches, structural discoveries, and mathematical validations dynamically.
- `src/tools/grounding.py`: Houses the embedding models and implements Reciprocal Rank Fusion (RRF) for hybrid search operations within the graph.
- `src/tools/discovery.py` & `src/tools/validation.py`: Contain the specialized graph analytics routines (finding shared methods, computing semantic vector bridges, Adamic-Adar calculation).

---

## 6. Packages and Libraries Used
- **User Interface & Visualization:**
  - `streamlit`, `streamlit.components.v1`: Frontend framework and component rendering.
  - `plotly`: Used for temporal charts and quantitative visualizations.
  - `pyvis`: Interactive HTML-based network graph visualization.
- **Graph & Mathematics:**
  - `networkx`: Core engine for creating, manipulating, and studying the structure of complex networks.
  - `numpy`: Numerical operations for matrix and vector calculations.
- **AI, Agents, and LLM Orchestration:**
  - `langchain`, `langchain-huggingface`, `langchain-community`: For prompt templating, tool binding, and LLM API interaction (specifically Qwen 2.5 via HuggingFace).
  - `langgraph`: To model the multi-agent reasoning flow as a state machine.
  - `pydantic`: For enforcing strict schemas in structured JSON data extraction.
- **Natural Language Processing & Embeddings:**
  - `sentence-transformers`: For generating dense semantic vector embeddings (`all-mpnet-base-v2`) and cross-encoder similarity scoring (`stsb-roberta-base`).
  - `spacy`, `scispacy` (`en_core_sci_sm`): For specialized scientific named entity recognition (NER) and sentence segmentation.
  - `rank-bm25`: For sparse lexical search capabilities.
  - `rapidfuzz`: To perform fast string matching and fuzzy logic during entity canonicalization.
  - `scikit-learn`: For calculating pairwise cosine similarities between high-dimensional vectors.
- **File Processing & Environment:**
  - `pypdf`: To parse and extract textual content from local PDF files.
  - `python-dotenv`: To load sensitive API keys from `.env` files.

---

## 7. Algorithms Used
The engine utilizes several advanced algorithmic approaches to process data and infer new scientific links:
- **Hybrid Search via Reciprocal Rank Fusion (RRF):**
  - **Algorithm:** Combines a Sparse Lexical Search (BM25Okapi) with a Dense Vector Search (Cosine Similarity). RRF calculates a combined score for each node using the formula: $RRF\_Score = \frac{1}{k + dense\_rank} + \frac{1}{k + sparse\_rank}$ (where $k=60$). This ensures high-quality retrieval combining keyword matching and contextual meaning.
- **Bidirectional Soft-Jaccard (Bipartite Semantic Matching):**
  - **Algorithm:** Maps concepts between two disparate papers using pairwise cosine similarity matrices. It calculates the maximum bidirectional alignment and derives a soft similarity score to inject semantic bridges when exact topological links fail.
- **Adamic-Adar Index:**
  - **Algorithm:** A topological link prediction metric implemented via NetworkX. It computes the probability of a hidden link based on the number of shared features (or neighbors) between two nodes, logarithmically weighted by the degree of those shared nodes.
- **Cross-Encoder Verification (STS-B):**
  - **Algorithm:** After dense retrieval, candidate pairs are passed through a HuggingFace cross-encoder (`stsb-roberta-base`) to accurately predict joint semantic relevance, scoring the interaction between pairs simultaneously.
- **Shortest Path and $k$-Hop Ego Extraction:**
  - **Algorithm:** Uses Breadth-First Search (BFS) / Dijkstra's shortest path underlying NetworkX to evaluate paths and retrieve isolated neighborhoods (ego graphs) within a parameterized radius ($k$-hops).
- **Fuzzy String Matching (Levenshtein Distance):**
  - **Algorithm:** Used during PDF ingestion (`rapidfuzz`) to map newly extracted triplet entities against existing canonical nodes in the graph if their similarity ratio exceeds a predefined threshold (85%).
