# slice_graph.py
import pickle
import networkx as nx

# 원래 그래프 파일에서 받아오기 
with open("nlp_papers_with_cid.gpickle", "rb") as f:
    G = pickle.load(f)
    
# louvain cluster id 필드가 0 인 노드들 저장
cluster_nodes = [n for n, d in G.nodes(data=True) if d.get("louvain_cluster_id") == 0]
# 위 리스트로 슬라이싱 -> 0번 클러스터
subgraph = G.subgraph(cluster_nodes).copy()

# 0번 클러스터 그래프를 아래 경로에 저장
with open("graph_cluster0.gpickle", "wb") as f:
    pickle.dump(subgraph, f)