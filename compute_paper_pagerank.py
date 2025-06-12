# =============================================
# File: compute_paper_pagerank.py
# Description:
#   This script reads a directed citation graph from 
#   citation_graph_with_cluster_v3.json and computes the PageRank 
#   for each paper node using citation edge weights.
#   The result includes metadata (title, year, authors, citationCount) 
#   and PageRank score, saved to papers_by_pagerank.json.
# =============================================


import networkx as nx
import json

# 1. Load the graph data
with open('citation_graph_with_cluster_v3.json', 'r') as f:
    graph_data = json.load(f)

G = nx.DiGraph()

# 2. Add nodes
for node in graph_data['nodes']:
    G.add_node(node['id'], 
               title=node.get('name', ''),
               year=node.get('year'),
               authors=node.get('authors'),
               citationCount=node.get('citationCount', 0))

# 3. Add edges
for edge in graph_data['links']:
    G.add_edge(edge['source'], edge['target'], 
               weight=edge.get('weight', 1.0))

# 4. Compute PageRank
pagerank_scores = nx.pagerank(G, weight='weight')

# 5. Merge scores into node data
ranked_papers = []
for node_id, score in pagerank_scores.items():
    data = G.nodes[node_id]
    ranked_papers.append({
        'id': node_id,
        'title': data['title'],
        'year': data['year'],
        'authors': data['authors'],
        'citationCount': data['citationCount'],
        'pagerank': score
    })

# 6. Sort by PageRank (descending)
ranked_papers.sort(key=lambda x: x['pagerank'], reverse=True)

# 7. Save as JSON
with open('papers_by_pagerank.json', 'w', encoding='utf-8') as f:
    json.dump(ranked_papers, f, ensure_ascii=False, indent=2)

print("✅ 저장 완료: papers_by_pagerank.json")
