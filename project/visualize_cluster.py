# slice_graph.py
import pickle
import networkx as nx

# 원래 그래프 파일에서 받아오기 
with open(r"nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)
    
# louvain cluster id 필드가 0 인 노드들 저장
cluster_nodes = [n for n, d in G.nodes(data=True) if d.get("louvain_cluster_id") == 0]
# 위 리스트로 슬라이싱 -> 0번 클러스터터
subgraph = G.subgraph(cluster_nodes).copy()

with open("graph_similarity_cluster00.gpickle", "wb") as f:
    pickle.dump(subgraph, f)
