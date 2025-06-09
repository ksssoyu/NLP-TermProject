import pickle
import networkx as nx
import pandas as pd

# ------------------------------
# 1. 그래프 불러오기
# ------------------------------
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

# ------------------------------
# 2. Eigenvector Centrality 계산
# ------------------------------
centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

# ------------------------------
# 3. DataFrame으로 정리
# ------------------------------
rows = []
for node_id, score in centrality.items():
    title = G.nodes[node_id].get("title", "")
    cluster = G.nodes[node_id].get("louvain_cluster_id", "")
    rows.append({
        "paper_id": node_id,
        "title": title,
        "eigenvector_centrality": score,
        "louvain_cluster_id": cluster
    })

df = pd.DataFrame(rows)
df = df.sort_values(by="eigenvector_centrality", ascending=False)

# ------------------------------
# 4. CSV로 저장
# ------------------------------
df.to_csv("eigenvector_centrality.csv", index=False, encoding="utf-8")
print("✅ eigenvector_centrality.csv 저장 완료")
