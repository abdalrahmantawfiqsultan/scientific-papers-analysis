# Hidden Scientific Links Discovery Engine (HSLDE) Documentation

## 1. Introduction
The **Hidden Scientific Links Discovery Engine (HSLDE)** is an advanced, AI-powered interactive platform designed to discover, reason about, and visualize implicit connections and methodological bridges between disparate scientific domains. The project leverages Large Language Models (LLMs), natural language processing (NLP), knowledge graphs (KG), and multi-agent workflows to synthesize novel scientific hypotheses from existing literature.

---

## 2. Functional Requirements
The system must fulfill the following functional capabilities:
- **Interactive Knowledge Graph Management:** Maintain an in-memory knowledge graph that stores scientific entities (Papers, Authors, Methods, Concepts, Datasets, Research Problems) and directed relationships between them.
- **Edge Provenance (NFR-12):** Every edge MUST carry strict provenance metadata: `source` (restricted to `extracted`, `algorithm`, `citation`, or `user`), `confidence` (float `[0.0, 1.0]`), and a `provenance` dictionary (`document_id`, `extraction_method`, `created_at`, `created_by`).
- **Interactive Graph Editor:** Allow users to directly manipulate the graph via a UI form. Includes creating/editing/deleting nodes, authoring custom edges (with `user` provenance), two-step safe deletion UX, file reading (FR-25), and web search (FR-28).
- **Agentic Autonomous Discovery:** Allow users to submit complex scientific inquiries. An autonomous pipeline executes Grounding, Structural Discovery, Validation, Self-Correction, and Synthesis phases. 
- **2D Graph Exploration:** Provide an interactive canvas allowing users to visualize the full graph, specific $k$-hop ego subgraphs, and shortest paths between entities.
- **Local Document Ingestion & Extraction:** Allow users to upload scientific papers in PDF format. The system persists the files to `data/uploads/`, routes them through either a fast PyMuPDF parser or a heavy Docling parser based on structure, extracts metadata, identifies dense scientific sentences (via spaCy), and leverages structured LLM outputs to extract entities and relationships.

---

## 3. Non-Functional Requirements
- **Performance & Context Management:** The system uses a 20,000-character NLP input window bound for spaCy processing to avoid performance bottlenecks, and truncates text to 15,000 characters before sending it to the LLM to preserve context limits.
- **Scalability & Responsiveness:** The agentic workflow streams updates progressively to the UI.
- **Resilience (NFR-08):** External API calls implement exponential backoff and retries (e.g., via the `requests` library adapter). The LLM extraction falls back gracefully to a "Partial Ingestion" state (metadata only) if the LLM provider is unreachable.
- **Aesthetics & Usability:** The dashboard features a restrained style (no glows/gradients) and sidebar navigation, enforcing strict UI design guidelines.

---

## 4. Interface Requirements
The user interface is a web-based dashboard built using Streamlit, comprising the following tabs:
- **Global Header/Telemetry:** Displays the application title alongside live metrics.
- **Research Agent (Tab 1):** Text input for inquiries, progressive LangGraph execution logs, and typewriter-effect synthesis.
- **Knowledge Graph (Tab 2):** PyVis HTML rendering of the graph subgraph, capable of panning, zooming, and node highlighting.
- **Local PDF Ingestion (Tab 3):** File uploader supporting PDFs. Progressive log window showing triage, Docling/PyMuPDF parsing, LLM extraction (Qwen/Groq/OpenAI/Gemini), and Graph canonicalization.
- **Graph Editor (Tab 4):** UI controls to Add/Edit/Delete nodes and relationships interactively.

---

## 5. Python Implementation Details
The application is structured into a modular Python codebase:

```text
sientificPapersAnalysis/
├── main.py                     # Application entry point to run Streamlit
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables (API keys: GROQ, OPENAI, GOOGLE, HUGGINGFACE)
├── benchmarks/                 # Automated benchmarking suite
│   ├── benchmark_router.py     # Parser accuracy and latency benchmarking
│   └── papers/                 # Test corpus directory
└── src/
    ├── ui/                     # Frontend user interface
    │   ├── app.py              # Main Streamlit dashboard layout and state
    │   ├── components.py       # Reusable UI components
    │   └── styles.py           # Custom CSS styling
    ├── graph/                  # Knowledge Graph management
    │   ├── in_memory_store.py  # NetworkX wrapper for the graph database (with strict provenance and type coercion)
    │   ├── visualizer.py       # PyVis HTML graph rendering
    │   └── db.py               # Database connections
    ├── agent/                  # Autonomous LLM workflow
    │   ├── workflow.py         # LangGraph state machine definition
    │   ├── nodes.py            # Centralized get_llm() router and workflow phase logic
    │   └── state.py            # TypedDict defining the agent's memory
    ├── ingestion/              # Ingestion pipeline
    │   ├── docling_parser.py   # Heavy multimodal parser for complex layouts
    │   ├── pymupdf_parser.py   # Fast text-based parser for born-digital PDFs
    │   ├── router.py           # Smart routing between parsers based on triage heuristics
    │   └── schema.py           # Pydantic schemas for extracted entities
    └── tools/                  # Analytical algorithms and capabilities
        ├── text_processing.py  # spaCy dense-sentence and entity extraction (NER), dateutil normalization
        ├── grounding.py        # Embeddings and BM25 RRF hybrid search
        ├── discovery.py        # Topological and semantic bridge discovery
        └── validation.py       # Link probability and temporal analysis
```

### Key Components
- `src/ui/app.py`: Coordinates event streams, the multi-tab layout, File Path isolation (FR-25), and handles `requests.exceptions.RequestException` to trigger "Partial Ingestion".
- `src/graph/in_memory_store.py`: `InMemoryKnowledgeGraph` wrapper over `networkx.MultiDiGraph`. Enforces strict `self._VALID_EDGE_SOURCES` on `add_relation`, and provides node/edge mutation primitives (`add_entity`, `edit_entity`, `delete_entity`, `delete_relation`).
- `src/agent/nodes.py`: Contains the `get_llm()` central router which dynamically selects from Groq, OpenAI, Gemini, or HuggingFace based on available environment variables.
- `src/ingestion/router.py`: Determines whether to use `DoclingIngestor` or `PyMuPDFParser`.
- `src/tools/text_processing.py`: Handles dense scientific sentence extraction and author/organization NER using `scispacy`.

---

## 6. Packages and Libraries Used
- **User Interface & Visualization:** `streamlit`, `plotly`, `pyvis`.
- **Graph & Mathematics:** `networkx`, `numpy`.
- **AI, Agents, and LLM Orchestration:** `langchain`, `langchain-groq`, `langchain-openai`, `langchain-google-genai`, `langchain-huggingface`, `langgraph`, `pydantic`.
- **Natural Language Processing & Embeddings:** `sentence-transformers`, `spacy`, `scispacy` (`en_core_sci_sm`), `rank-bm25`, `rapidfuzz`, `scikit-learn`.
- **File Processing & Environment:** `docling`, `pymupdf` (`fitz`), `python-dateutil`, `requests`, `python-dotenv`.

---

## 7. Algorithms Used
- **Hybrid Search via Reciprocal Rank Fusion (RRF)**
- **Bidirectional Soft-Jaccard (Bipartite Semantic Matching)**
- **Adamic-Adar Index**
- **Cross-Encoder Verification (STS-B)**
- **Shortest Path and $k$-Hop Ego Extraction**
- **Fuzzy String Matching (Levenshtein Distance)**
