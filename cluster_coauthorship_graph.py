# ============================================================================
# File: cluster_coauthorship_graph.py
# Description:
#   This script performs Leiden clustering on a co-authorship graph and analyzes
#   both internal collaboration and inter-cluster interactions.
#
# Functionality:
#   1. Loads a JSON-formatted undirected co-authorship graph with weights
#   2. Applies Leiden clustering to group authors by collaboration communities
#   3. Outputs:
#      - Cluster assignment for each author
#      - Top 10 clusters (by normalized internal collaboration strength)
#      - Top 5 authors per top cluster (by internal edge weight)
#      - Annotated graph with cluster_id added to nodes
#      - Inter-cluster connectivity summary (Top 2 connected clusters)
#
# Input:  coauthorship_graph_with_year.json
# Outputs:
#   - coauthorship_clusters.json              (cluster memberships)
#   - top10_cluster_top5_authors.json         (top authors in key clusters)
#   - coauthorship_graph_with_year_cluster.json (graph with cluster_id)
#   - external_cluster_links.json             (top 2 inter-cluster links)
#
# Note:
#   - Resolution parameter (default: 0.5) can be tuned for cluster granularity
# ============================================================================


import json
import igraph as ig
import leidenalg
from collections import defaultdict

# ===== 설정 =====
input_path = "coauthorship_graph_with_year.json"
cluster_output_path = "coauthorship_clusters.json"
top10_output_path = "top10_cluster_top5_authors.json"
annotated_graph_output = "coauthorship_graph_with_year_cluster.json"
external_cluster_links_output = "external_cluster_links.json"
resolution = 0.5  # 클러스터 크기 조절용 파라미터

# ===== 1. JSON 로드 =====
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]
links = data["links"]

# ===== 2. 노드 ID 매핑 =====
node_ids = [node["id"] for node in nodes]
id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

# ===== 3. 엣지 생성 =====
edges = []
weights = []
for link in links:
    src, tgt = link["source"], link["target"]
    if src in id_to_index and tgt in id_to_index:
        edges.append((id_to_index[src], id_to_index[tgt]))
        weights.append(float(link.get("weight", 1.0)))

# ===== 4. igraph 그래프 생성 =====
g = ig.Graph(n=len(node_ids), edges=edges, directed=False)
g.vs["name"] = node_ids
g.es["weight"] = weights

# ===== 5. Leiden 클러스터링 수행 =====
partition = leidenalg.find_partition(
    g,
    leidenalg.RBConfigurationVertexPartition,
    weights=g.es["weight"],
    resolution_parameter=resolution
)

# ===== 6. 클러스터 결과 저장 =====
cluster_details = defaultdict(list)
node_to_cluster = {}

for idx, cluster_id in enumerate(partition.membership):
    node_id = g.vs[idx]["name"]
    cluster_details[cluster_id].append({"id": node_id})
    node_to_cluster[node_id] = cluster_id

with open(cluster_output_path, "w", encoding="utf-8") as f:
    json.dump({
        "num_clusters": len(cluster_details),
        "cluster_details": cluster_details
    }, f, ensure_ascii=False, indent=2)

print(f"✅ 클러스터링 완료: {cluster_output_path}")

# ===== 7. 클러스터별 내부 강도 계산 + 외부 클러스터 연관도 분석 =====
cluster_nodes = {int(k): set(p["id"] for p in v) for k, v in cluster_details.items()}
cluster_strength = defaultdict(float)
external_cluster_links = defaultdict(dict)

for link in links:
    src, tgt = link["source"], link["target"]
    w = float(link.get("weight", 1.0))
    src_cid = node_to_cluster.get(src)
    tgt_cid = node_to_cluster.get(tgt)

    if src_cid is None or tgt_cid is None:
        continue
    if src_cid == tgt_cid:
        cluster_strength[src_cid] += w
    else:
        external_cluster_links[src_cid][tgt_cid] = external_cluster_links[src_cid].get(tgt_cid, 0) + w
        external_cluster_links[tgt_cid][src_cid] = external_cluster_links[tgt_cid].get(src_cid, 0) + w

normalized_cluster_score = {
    cid: cluster_strength[cid] / max(len(cluster_nodes[cid]), 1)
    for cid in cluster_nodes
}

# ===== 8. Top 10 클러스터 선정 =====
top10_clusters = sorted(normalized_cluster_score.items(), key=lambda x: x[1], reverse=True)[:10]
top10_ids = [cid for cid, _ in top10_clusters]
print("🏆 Top 10 클러스터 ID:", top10_ids)

# ===== 9. Top 5 저자 선정 =====
top5_authors_per_cluster = {}

for cid in top10_ids:
    node_set = cluster_nodes[cid]
    author_score = defaultdict(float)

    for link in links:
        src, tgt = link["source"], link["target"]
        w = float(link.get("weight", 1.0))
        if src in node_set and tgt in node_set:
            author_score[src] += w
            author_score[tgt] += w

    top_authors = sorted(author_score.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_authors_per_cluster[cid] = [
        {"author": author, "total_internal_collab_weight": weight}
        for author, weight in top_authors
    ]

with open(top10_output_path, "w", encoding="utf-8") as f:
    json.dump(top5_authors_per_cluster, f, ensure_ascii=False, indent=2)
print(f"📁 Top 10 클러스터별 협업 상위 5명 author 저장 완료: {top10_output_path}")

# ===== 10. 클러스터 ID 포함 그래프 저장 =====
annotated_nodes = []
for node in nodes:
    nid = node["id"]
    new_node = dict(node)
    new_node["cluster_id"] = node_to_cluster.get(nid, -1)
    annotated_nodes.append(new_node)

annotated_graph = {
    "nodes": annotated_nodes,
    "links": links
}

with open(annotated_graph_output, "w", encoding="utf-8") as f:
    json.dump(annotated_graph, f, ensure_ascii=False, indent=2)
print(f"📁 클러스터 ID 추가된 그래프 저장 완료: {annotated_graph_output}")

# ===== 11. 외부 클러스터 연관도 Top2 저장 =====
external_top2 = {}
for cid, neighbors in external_cluster_links.items():
    sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:2]
    external_top2[cid] = [{"cluster_id": nid, "total_link_weight": w} for nid, w in sorted_neighbors]

with open(external_cluster_links_output, "w", encoding="utf-8") as f:
    json.dump(external_top2, f, ensure_ascii=False, indent=2)
print(f"📁 외부 클러스터와의 연관도 Top2 저장 완료: {external_cluster_links_output}")
