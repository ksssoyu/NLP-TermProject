import networkx as nx
import pandas as pd
import pickle
from collections import defaultdict

# -------------------------------
# 1. 그래프 및 클러스터 로딩
# -------------------------------
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

df_cluster = pd.read_csv("paper_clusters.csv")
cluster_map = dict(zip(df_cluster["paper_id"], df_cluster["community_id"]))

# -------------------------------
# 2. inter-cluster betweenness 계산
# -------------------------------
inter_cluster_betweenness = defaultdict(float)
nodes = list(G.nodes)

for i, source in enumerate(nodes):
    source_cluster = cluster_map.get(source)
    if source_cluster is None:
        continue

    # 최단 경로 구하기
    try:
        lengths, paths = nx.single_source_shortest_path_length(G, source), nx.single_source_shortest_path(G, source)
    except nx.NetworkXError:
        continue

    for target, path in paths.items():
        if source == target:
            continue
        target_cluster = cluster_map.get(target)
        if target_cluster is None or target_cluster == source_cluster:
            continue  # 내부 경로는 무시

        # 중간 노드들에 점수 분배
        intermediates = path[1:-1]  # source와 target은 제외
        for v in intermediates:
            inter_cluster_betweenness[v] += 1

    if i % 500 == 0:
        print(f"🔄 진행 중: {i}/{len(nodes)}")

# -------------------------------
# 3. 결과 정렬 및 저장
# -------------------------------
result_df = pd.DataFrame([
    (node, inter_cluster_betweenness[node])
    for node in inter_cluster_betweenness
], columns=["node", "inter_cluster_betweenness"])

result_df = result_df.sort_values(by="inter_cluster_betweenness", ascending=False)
result_df.to_csv("inter_cluster_betweenness.csv", index=False)

print("✅ inter_cluster_betweenness.csv 저장 완료!")
