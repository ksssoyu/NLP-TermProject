import pickle
import networkx as nx
import pandas as pd
import os

# ------------------------------
# 1. 그래프 불러오기
# ------------------------------
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

# ------------------------------
# 2. 클러스터 ID 목록 추출
# ------------------------------
cluster_ids = set(nx.get_node_attributes(G, "louvain_cluster_id").values())

# ------------------------------
# 3. 출력 폴더 생성
# ------------------------------
output_dir = "cluster_centrality"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 4. 클러스터별 Centrality 계산 및 저장
# ------------------------------
for cluster_id in sorted(cluster_ids):
    nodes_in_cluster = [n for n, d in G.nodes(data=True) if d.get("louvain_cluster_id") == cluster_id]
    subgraph = G.subgraph(nodes_in_cluster)
    
    if len(subgraph) < 2:
        continue

    try:
        centrality = nx.eigenvector_centrality(subgraph, weight='weight', max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print(f"⚠️ 클러스터 {cluster_id}에서 수렴 실패")
        continue

    rows = []
    for node_id, score in centrality.items():
        title = G.nodes[node_id].get("title", "")
        rows.append({
            "paper_id": node_id,
            "title": title,
            "eigenvector_centrality": score
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="eigenvector_centrality", ascending=False)
    df.to_csv(f"{output_dir}/cluster_{cluster_id}.csv", index=False, encoding="utf-8")

print(f"✅ 클러스터별 centrality 결과를 '{output_dir}/' 폴더에 저장 완료!")
