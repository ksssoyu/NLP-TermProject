import pickle
import networkx as nx

# 그래프 불러오기
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

internal_weights = []
external_weights = []

for u, v, data in G.edges(data=True):
    cluster_u = G.nodes[u].get("louvain_cluster_id")
    cluster_v = G.nodes[v].get("louvain_cluster_id")
    weight = data.get("weight", 0)

    if cluster_u is None or cluster_v is None:
        continue

    if cluster_u == cluster_v:
        internal_weights.append(weight)
    else:
        external_weights.append(weight)

# 평균 계산
avg_internal = sum(internal_weights) / len(internal_weights)
avg_external = sum(external_weights) / len(external_weights)

print(f"📊 내부 유사도 평균: {avg_internal:.4f}")
print(f"📉 외부 유사도 평균: {avg_external:.4f}")
print(f"✅ 내부/외부 비율: {(avg_internal / avg_external):.2f}배")
