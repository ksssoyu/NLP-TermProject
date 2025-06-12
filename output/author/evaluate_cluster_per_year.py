import json
import os
import networkx as nx
from networkx.algorithms.community.quality import modularity
import pandas as pd

# 📁 경로 설정
GRAPH_PATH = "coauthorship_graph_with_year.json"
CLUSTER_DIR = "coauthorship_cluster_by_year"

# 📥 공동저자 그래프 로드
with open(GRAPH_PATH, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

all_links = graph_data["links"]

# ✅ 연도 윈도우 필터 함수
def in_window(link_year, window_str):
    try:
        y = int(link_year)
        start, end = map(int, window_str.split("_"))
        return start <= y <= end
    except:
        return False

# ✅ performance 직접 구현
def custom_performance(G, communities):
    node_community = {}
    for c in communities:
        for node in c:
            node_community[node] = tuple(c)

    correct = 0
    total = 0
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            same_cluster = node_community.get(u) == node_community.get(v)
            connected = G.has_edge(u, v)
            if (same_cluster and connected) or (not same_cluster and not connected):
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

# ✅ coverage 직접 구현
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

# ✅ conductance 계산
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

# 📊 결과 저장
results = []

# 🔁 연도별 클러스터 평가
for filename in sorted(os.listdir(CLUSTER_DIR)):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(CLUSTER_DIR, filename), "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    window = cluster_data["window"]
    cluster_details = cluster_data["cluster_details"]

    # 그래프 생성
    G = nx.Graph()
    for link in all_links:
        if in_window(link["years"], window):
            G.add_edge(link["source"], link["target"], weight=link.get("weight", 1.0))

    print(f"{window}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 클러스터 목록
    communities = []
    for members in cluster_details.values():
        node_list = [m["id"] for m in members if m["id"] in G]
        if node_list:
            communities.append(node_list)

    # 서브그래프 생성
    cluster_nodes = set()
    for c in communities:
        cluster_nodes.update(c)
    G_sub = G.subgraph(cluster_nodes)

    # 지표 계산
    mod = modularity(G_sub, communities) if communities else 0
    cov = custom_coverage(G_sub, communities) if communities else 0
    perf = custom_performance(G_sub, communities) if communities else 0
    conds = [compute_conductance(G_sub, c) for c in communities if len(c) > 1]
    avg_cond = sum(conds) / len(conds) if conds else 0

    results.append({
        "window": window,
        "modularity": round(mod, 4),
        "coverage": round(cov, 4),
        "performance": round(perf, 4),
        "avg_conductance": round(avg_cond, 4),
        "num_clusters": len(communities),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges()
    })

# ✅ 출력
df = pd.DataFrame(results)
print(df.to_string(index=False))

# ✅ 원하면 CSV로 저장
# df.to_csv("cluster_evaluation_results.csv", index=False)
