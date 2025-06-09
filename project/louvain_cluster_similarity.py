import pickle
import networkx as nx
import community as community_louvain  # pip install python-louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict, Counter
import csv
import pandas as pd

# ------------------------------
# 1. 그래프 불러오기
# ------------------------------
with open('nlp_similarity_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

print(f"📦 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")

# ------------------------------
# 2. Louvain 클러스터링 수행
# ------------------------------
partition = community_louvain.best_partition(G, weight='weight')

# partition: 딕셔너리 {node_id: community_id}
num_clusters = len(set(partition.values()))
print(f"🔍 Louvain으로 {num_clusters}개의 클러스터 탐지됨")

# ➕ 클러스터 ID를 노드 속성으로 저장
for node, cluster_id in partition.items():
    G.nodes[node]["louvain_cluster_id"] = cluster_id

# ------------------------------
# 3. 결과 시각화
# ------------------------------
pos = nx.spring_layout(G, weight='distance', seed=42)
cmap = cm.get_cmap('tab20', num_clusters)

plt.figure(figsize=(13, 13))
for comm_id in set(partition.values()):
    node_list = [node for node in G.nodes if partition[node] == comm_id]
    nx.draw_networkx_nodes(G, pos,
                           nodelist=node_list,
                           node_size=50,
                           node_color=[cmap(comm_id)],
                           alpha=0.8)

nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.3)
plt.title("Louvain Clustering of NLP Paper Similarity Graph")
plt.axis('off')
plt.tight_layout()
plt.show()

# ------------------------------
# 4. 클러스터 결과 CSV 저장
# ------------------------------
partition_df = pd.DataFrame(list(partition.items()), columns=['paper_id', 'community_id'])
partition_df.to_csv("paper_clusters.csv", index=False)

# ➕ 그래프도 저장
with open("nlp_similarity_graph_with_cluster.gpickle", "wb") as f:
    pickle.dump(G, f)
print("✅ 클러스터 ID가 포함된 그래프를 nlp_similarity_graph_with_cluster.gpickle 으로 저장했습니다.")

# ------------------------------
# 5. 클러스터별 주요 개념 저장
# ------------------------------
def save_top_concepts_by_cluster(G, top_n=10, output_file="cluster_concepts.csv"):
    cluster_concepts = defaultdict(list)

    for _, data in G.nodes(data=True):
        cluster_id = data.get("louvain_cluster_id")
        concepts_str = data.get("concepts", "")
        if cluster_id is None or not concepts_str:
            continue
        concepts = [c.strip().lower() for c in concepts_str.split(";") if c.strip()]
        cluster_concepts[cluster_id].extend(concepts)

    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "concept", "count"])
        for cluster_id, concept_list in cluster_concepts.items():
            counter = Counter(concept_list)
            for concept, count in counter.most_common(top_n):
                writer.writerow([cluster_id, concept, count])

save_top_concepts_by_cluster(G, top_n=10)
print("✅ 클러스터별 주요 concept를 cluster_concepts.csv로 저장했습니다.")
