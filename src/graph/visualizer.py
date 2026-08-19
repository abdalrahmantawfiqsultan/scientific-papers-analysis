import os
import networkx as nx
from pyvis.network import Network

# Color palette mapped by entity type
COLOR_PALETTE = {
    "Paper": {"background": "#3B82F6", "border": "#1D4ED8", "highlight": "#60A5FA"},     # Blue
    "Method": {"background": "#10B981", "border": "#047857", "highlight": "#34D399"},    # Green / Teal
    "Concept": {"background": "#F59E0B", "border": "#B45309", "highlight": "#FBBF24"},   # Amber
    "Field": {"background": "#8B5CF6", "border": "#6D28D9", "highlight": "#A78BFA"},     # Purple
    "Author": {"background": "#EC4899", "border": "#BE185D", "highlight": "#F472B6"},    # Pink
    "Dataset": {"background": "#06B6D4", "border": "#0E7490", "highlight": "#22D3EE"},   # Cyan
    "Default": {"background": "#6B7280", "border": "#374151", "highlight": "#9CA3AF"}    # Gray
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
            bgcolor="#0F172A",          # Slate dark theme
            font_color="#F8FAFC"
        )

        if not graph.nodes:
            return net

        # Calculate degree centrality to scale node sizes dynamically
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1

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
            <div style="font-family: Arial; padding: 6px; font-size: 13px;">
                <b style="color: {palette['background']};">[{label}]</b> <b>{data.get('name', node_id)}</b><br/>
                <hr style="margin: 4px 0; border: 0.5px solid #475569;"/>
                <b>Connections:</b> {node_deg}<br/>
                {f"<b>Year:</b> {data.get('year')}<br/>" if 'year' in data else ""}
                {f"<b>Description:</b> {data.get('description')}<br/>" if 'description' in data else ""}
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
                font={"size": 14, "color": "#FFFFFF", "face": "Helvetica"}
            )

        # 2. Add Edges with Relationship Labels
        for source, target, key, data in graph.edges(keys=True, data=True):
            relation = data.get("relation", key)
            
            # Highlight AI-discovered semantic bridges with glowing red/orange
            if relation == "AGENT_DISCOVERY":
                edge_color = "#EF4444"
                width = 4
                dash = True
            else:
                edge_color = "#64748B"
                width = 1
                dash = False

            net.add_edge(
                source, target,
                label=relation,
                title=f"Relation: {relation} | {data.get('rationale', '')}",
                color=edge_color,
                width=width,
                dashes=dash,
                arrows="to",
                font={"size": 10, "color": "#F8FAFC", "align": "middle"}
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
