import json
import networkx as nx
import community as community_louvain  # Louvain 알고리즘
import matplotlib.pyplot as plt

# 1. JSON 로드
with open("citation_graph_with_cluster_v2.json", "r") as f:
    data = json.load(f)

nodes = data["nodes"]
links = data["links"]

# 2. NetworkX DiGraph 생성
G = nx.DiGraph()
for node in nodes:
    G.add_node(node["id"])

for link in links:
    src = link["source"]
    tgt = link["target"]
    weight = link.get("weight", 1.0)
    G.add_edge(src, tgt, weight=weight)

# 3. 무방향 그래프로 변환 (Louvain은 undirected 필요)
G_undirected = G.to_undirected()

# 4. Louvain 클러스터링 수행
partition = community_louvain.best_partition(G_undirected, weight='weight')

# 5. 결과 예시 출력
print("📦 총 클러스터 수:", len(set(partition.values())))
for pid, cluster_id in list(partition.items())[:10]:
    print(f"{pid[:8]}... → Cluster {cluster_id}")
