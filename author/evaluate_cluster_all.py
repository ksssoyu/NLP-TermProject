import json
import networkx as nx
from networkx.algorithms.community.quality import modularity
import pandas as pd

# 🔧 경로 설정
GRAPH_PATH = "coauthorship_graph_with_year_cluster.json"

# 📥 그래프 데이터 로드
with open(GRAPH_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]
links = data["links"]

# ✅ 그래프 생성
G = nx.Graph()
for node in nodes:
    G.add_node(node["id"], cluster_id=node.get("cluster_id"))

for link in links:
    G.add_edge(link["source"], link["target"], weight=link.get("weight", 1.0))

print(f"전체 그래프: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ✅ 커뮤니티 목록 생성 (cluster_id 기준)
from collections import defaultdict

cluster_map = defaultdict(list)
for node in G.nodes(data=True):
    cid = node[1].get("cluster_id")
    if cid is not None:
        cluster_map[cid].append(node[0])

communities = list(cluster_map.values())

# ✅ 클러스터 노드만으로 서브그래프
cluster_nodes = set().union(*communities)
G_sub = G.subgraph(cluster_nodes)

# ✅ 평가 지표 함수들
def fast_performance(G, communities):
    # 노드 → 클러스터 ID 매핑
    node_to_comm = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = idx

    same_comm_edges = 0
    total_edges = 0

    for u in G.nodes():
        for v in G.nodes():
            if u == v:
                continue
            total_edges += 1
            same_cluster = node_to_comm.get(u) == node_to_comm.get(v)
            connected = G.has_edge(u, v)
            if (same_cluster and connected) or (not same_cluster and not connected):
                same_comm_edges += 1

    return same_comm_edges / total_edges if total_edges > 0 else 0

def custom_coverage(G, communities):
    intra_edges = 0
    total_edges = G.number_of_edges()
    seen_edges = set()
    
    for community in communities:
        for u in community:
            for v in G.neighbors(u):
                if v in community and (v, u) not in seen_edges:
                    intra_edges += 1
                    seen_edges.add((u, v))
    return intra_edges / total_edges if total_edges > 0 else 0

def compute_conductance(G, community):
    S = set(community)
    notS = set(G.nodes()) - S
    cut_edges = 0
    vol_S = 0
    vol_notS = 0
    for u in G.nodes():
        for v in G.neighbors(u):
            w = G[u][v].get("weight", 1)
            if u in S:
                vol_S += w
            else:
                vol_notS += w
            if (u in S and v in notS) or (u in notS and v in S):
                cut_edges += w
    denom = min(vol_S, vol_notS)
    return cut_edges / denom if denom > 0 else 0

# ✅ 지표 계산
mod = modularity(G_sub, communities)
cov = custom_coverage(G_sub, communities)
#perf = fast_performance(G_sub, communities)

conds = [compute_conductance(G_sub, c) for c in communities if len(c) > 1]

avg_cond = sum(conds) / len(conds) if conds else 0

# ✅ 출력
print("\n📊 전체 클러스터 평가 지표")
print(f"Modularity        : {mod:.4f}")
print(f"Coverage          : {cov:.4f}")
#print(f"Performance       : {perf:.4f}")
print(f"Avg Conductance   : {avg_cond:.4f}")
print(f"Num Clusters      : {len(communities)}")
print(f"Num Nodes         : {G_sub.number_of_nodes()}")
print(f"Num Edges         : {G_sub.number_of_edges()}")
