"""
Convert GraphML Co-authorship Graph to JSON Format

This script converts a co-authorship graph saved in GraphML format into a JSON format
that is suitable for web-based visualization or downstream analysis.

Features:
- Loads an undirected GraphML file (produced by NetworkX)
- Converts nodes and edges to JSON-compatible dictionaries
- Preserves edge weights and collaboration years
- Outputs a JSON file with 'nodes' and 'links' keys

Input:
- coauthorship_graph_with_year.graphml: GraphML file containing author nodes and edges with weights and years

Output:
- coauthorship_graph_with_year.json: JSON representation of the graph, with structure:
    {
      "nodes": [ { "id": "Author Name" }, ... ],
      "links": [ { "source": "Author A", "target": "Author B", "weight": 2.0, "years": "2018,2019" }, ... ]
    }

Usage:
- Run directly: `python convert_graphml_to_json.py`
"""


import networkx as nx
import json

# Load the existing GraphML file
graphml_file = "coauthorship_graph_with_year.graphml"
G = nx.read_graphml(graphml_file)

# Convert the GraphML to a JSON format suitable for analysis
nodes = []
for node in G.nodes():
    nodes.append({"id": node})

edges = []
for u, v, data in G.edges(data=True):
    edge = {
        "source": u,
        "target": v,
        "weight": float(data.get("weight", 1.0)),
        "years": data.get("years", "")
    }
    edges.append(edge)

graph_json = {
    "nodes": nodes,
    "links": edges
}

# Save the result to JSON
output_json_file = "coauthorship_graph_with_year.json"
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(graph_json, f, ensure_ascii=False, indent=2)