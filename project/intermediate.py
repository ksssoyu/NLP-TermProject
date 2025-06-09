#클러스터 경계에 있는 노드들을 추출하는 코드. 내부 클러스터 노드와 외부 클러스터 노드의 비율을 계산하여 외부 클러스터 노드가 40% 이상인 경우를 애매한 노드로 간주

import pandas as pd
import networkx as nx
import pickle

# 1. 그래프 로드
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

# 2. 클러스터 정보 로드
df_cluster = pd.read_csv("paper_clusters.csv")
cluster_map = dict(zip(df_cluster["paper_id"], df_cluster["community_id"]))

# 3. 제목 정보 로드
df_meta = pd.read_csv("merged_nlp_papers_for_enrichment.csv")
title_map = dict(zip(df_meta["paperId"], df_meta["title"]))

# 4. 애매한 노드 추출
ambiguous_nodes = []

for node in G.nodes:
    if node not in cluster_map:
        continue

    my_cluster = cluster_map[node]
    internal = 0
    external = 0

    for neighbor in G.neighbors(node):
        neighbor_cluster = cluster_map.get(neighbor)
        if neighbor_cluster is None:
            continue
        if neighbor_cluster == my_cluster:
            internal += 1
        else:
            external += 1

    total = internal + external
    if total == 0:
        continue

    external_ratio = external / total
    if external_ratio >= 0.4:
        ambiguous_nodes.append((
            node,
            title_map.get(node, "UNKNOWN TITLE"),
            cluster_map[node],  # ← 클러스터 ID 추가
            internal,
            external,
            round(external_ratio, 3)
        ))

# 5. 결과 저장
ambiguous_df = pd.DataFrame(ambiguous_nodes, columns=[
    "paper_id", "title", "cluster_id","internal_edges", "external_edges", "external_ratio"
])

ambiguous_df.to_csv("ambiguous_nodes.csv", index=False)
print("✅ ambiguous_nodes.csv 저장 완료!")
