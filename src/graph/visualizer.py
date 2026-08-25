import os
import networkx as nx
from pyvis.network import Network

# Color palette mapped by entity type — indigo-anchored with distinct hues per type
COLOR_PALETTE = {
    "Paper":           {"background": "#4F46E5", "border": "#3730A3", "highlight": "#818CF8"},     # Indigo
    "Author":          {"background": "#2563EB", "border": "#1D4ED8", "highlight": "#60A5FA"},     # Blue
    "Method":          {"background": "#8B5CF6", "border": "#6D28D9", "highlight": "#A78BFA"},     # Violet
    "Dataset":         {"background": "#0D9488", "border": "#0F766E", "highlight": "#2DD4BF"},     # Teal
    "Concept":         {"background": "#D97706", "border": "#B45309", "highlight": "#FBBF24"},     # Amber
    "Metric":          {"background": "#0891B2", "border": "#0E7490", "highlight": "#22D3EE"},     # Cyan
    "ResearchProblem": {"background": "#DC2626", "border": "#B91C1C", "highlight": "#F87171"},     # Red
    "Result":          {"background": "#9333EA", "border": "#7E22CE", "highlight": "#C084FC"},     # Purple
    "Organization":    {"background": "#EA580C", "border": "#C2410C", "highlight": "#FB923C"},     # Orange
    "Field":           {"background": "#475569", "border": "#334155", "highlight": "#94A3B8"},     # Slate
    "Default":         {"background": "#475569", "border": "#334155", "highlight": "#94A3B8"},     # Slate
}

class GraphVisualizer2D:
    def __init__(self, height: str = "750px", width: str = "100%"):
        self.height = height
        self.width = width

    def build_network(
        self, 
        graph: nx.MultiDiGraph, 
        center_highlight: str = None, 
        physics_enabled: bool = True
    ) -> Network:
        """Construct an optimized interactive PyVis Network."""
        
        net = Network(
            height=self.height, 
            width=self.width, 
            directed=True, 
            notebook=False, 
            bgcolor="#0C1222",          # Deep navy — matches app background
            font_color="#E2E8F0"
        )

        if not graph.nodes:
            return net

        # Calculate degree centrality to scale node sizes dynamically
        degrees = dict(graph.degree())
        max_degree = max(1, max(degrees.values()) if degrees else 1)

        # 1. Add Nodes with Custom Styling & Tooltips
        for node_id, data in graph.nodes(data=True):
            label = data.get("label", "Default")
            palette = COLOR_PALETTE.get(label, COLOR_PALETTE["Default"])
            
            # Dynamic node radius (scale between 15px and 45px)
            node_deg = degrees.get(node_id, 1)
            size = 15 + (node_deg / max_degree) * 30
            
            # Highlight center node if inspecting an ego-graph
            if center_node := center_highlight:
                if node_id == center_node:
                    size = 50

            # Rich HTML Tooltip
            tooltip = f"""
            <div style="font-family: 'Inter', Arial, sans-serif; padding: 10px; font-size: 13px; background: #1E293B; border-radius: 8px; border: 1px solid #334155; max-width: 300px;">
                <div style="color: {palette['background']}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">{label}</div>
                <div style="font-weight: 700; font-size: 15px; color: #F1F5F9; margin-bottom: 6px;">{data.get('name', node_id)}</div>
                <hr style="margin: 6px 0; border: 0.5px solid #334155;"/>
                <span style="color: #94A3B8;">Connections:</span> <b style="color: #A5B4FC;">{node_deg}</b><br/>
                {f'<span style="color: #94A3B8;">Year:</span> <b style="color: #FCD34D;">{data.get("year")}</b><br/>' if 'year' in data else ''}
                {f'<div style="color: #94A3B8; margin-top: 4px; font-size: 12px; line-height: 1.4;">{str(data.get("description", ""))[:150]}</div>' if 'description' in data else ''}
            </div>
            """

            # Truncate label to prevent massive overlapping text blobs
            raw_name = data.get("name", node_id)
            display_label = raw_name if len(raw_name) <= 25 else raw_name[:22] + "..."

            net.add_node(
                node_id,
                label=display_label,
                title=tooltip,
                size=size,
                color={
                    "background": palette["background"],
                    "border": palette["border"],
                    "highlight": {"background": palette["highlight"], "border": "#FFFFFF"}
                },
                borderWidth=2,
                font={"size": 13, "color": "#E2E8F0", "face": "Inter, Helvetica, Arial"}
            )

        # 2. Add Edges with Relationship Labels
        for source, target, key, data in graph.edges(keys=True, data=True):
            relation = data.get("relation", key)
            
            confidence = data.get("confidence", "")
            conf_str = f" · {confidence:.2f} confidence" if isinstance(confidence, float) else ""
            
            if data.get("source") == "algorithm":
                method = data.get("provenance", {}).get("extraction_method", "semantic similarity")
                edge_color = "#6366F1" # Muted indigo
                width = 2
                dash = True
                hover_title = f"◆ Inferred relationship\n{relation}{conf_str}\nMethod: {method}"
            else:
                doc = data.get("provenance", {}).get("document_id", "Unknown")
                edge_color = "#475569"
                width = 1.5
                dash = False
                hover_title = f"✓ Extracted\n{relation}\nSource: {doc}"

            net.add_edge(
                source, target,
                label=relation,
                title=hover_title,
                color=edge_color,
                width=width,
                dashes=dash,
                arrows="to",
                font={"size": 9, "color": "#94A3B8", "align": "middle"}
            )

        # 3. Apply Optimized Physics (Barnes-Hut algorithm with smooth damping)
        options = {
            "nodes": {"shape": "dot"},
            "edges": {
                # Use dynamic smoothing so parallel edges curve away from each other
                "smooth": {"type": "dynamic"}
            },
            "physics": {
                "enabled": physics_enabled,
                "barnesHut": {
                    "gravitationalConstant": -15000,
                    "centralGravity": 0.2,
                    "springLength": 250,      # Increased to spread nodes apart
                    "springConstant": 0.03,
                    "damping": 0.09,
                    "avoidOverlap": 1.0       # Maximize overlap prevention
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 150,     # Stop simulating after 150 steps for instant responsiveness
                    "updateInterval": 25
                }
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True, # Zoom and pan controls
                "keyboard": True,
                "tooltipDelay": 100
            }
        }
        
        import json
        net.set_options(f"""var options = {json.dumps(options)}""")
        return net

    def export_html(self, graph: nx.MultiDiGraph, output_path: str = "graph_view.html", center_node: str = None) -> str:
        """Export the graph directly to a standalone, browser-ready HTML file."""
        net = self.build_network(graph, center_highlight=center_node)
        net.save_graph(output_path)
        return os.path.abspath(output_path)

if __name__ == "__main__":
    import webbrowser
    
    print("Testing the GraphVisualizer2D...")
    # Create a small dummy graph to test the visualizer
    dummy_graph = nx.MultiDiGraph()
    dummy_graph.add_node("Paper:1", label="Paper", name="Test Paper", year=2026)
    dummy_graph.add_node("Method:1", label="Method", name="Test Method")
    dummy_graph.add_edge("Paper:1", "Method:1", relation="USES_METHOD")
    
    # Initialize visualizer and export
    vis = GraphVisualizer2D()
    out_path = vis.export_html(dummy_graph, output_path="test_vis.html")
    print(f"Visualizer test successful! Opening {out_path}")
    webbrowser.open_new(f"file://{out_path}")
