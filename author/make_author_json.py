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