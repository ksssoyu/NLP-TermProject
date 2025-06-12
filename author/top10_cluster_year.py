import os
import json
from collections import defaultdict

# 설정
GRAPH_PATH = "coauthorship_graph_with_year.json"
CLUSTER_DIR = "coauthorship_cluster_by_year"
OUTPUT_DIR = "top10_clusters_by_strength"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 전체 그래프 로딩
with open(GRAPH_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
all_links = data["links"]

# 전체 링크를 빠르게 탐색하기 위해 딕셔너리화
edge_weight_lookup = defaultdict(float)
for link in all_links:
    key = tuple(sorted([link["source"], link["target"]]))
    edge_weight_lookup[key] += float(link.get("weight", 1.0))

# 클러스터 파일 순회
for fname in sorted(os.listdir(CLUSTER_DIR)):
    if not fname.endswith(".json"):
        continue

    cluster_path = os.path.join(CLUSTER_DIR, fname)
    with open(cluster_path, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    cluster_details = cluster_data["cluster_details"]
    cluster_nodes = {
        int(cid): set(entry["id"] for entry in members)
        for cid, members in cluster_details.items()
    }

    # 클러스터별 내부 엣지 weight 합 계산
    cluster_strength = defaultdict(float)
    for cid, nodes in cluster_nodes.items():
        nodes_list = list(nodes)
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                key = tuple(sorted([nodes_list[i], nodes_list[j]]))
                cluster_strength[cid] += edge_weight_lookup.get(key, 0.0)

    # 노드 수로 정규화
    normalized_score = {
        cid: cluster_strength[cid] / max(len(cluster_nodes[cid]), 1)
        for cid in cluster_nodes
    }

    # Top 10 클러스터 선택
    top10 = sorted(normalized_score.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_ids = {cid for cid, _ in top10}

    # 필터링하여 새 파일 저장
    top_details = {
        str(cid): cluster_details[str(cid)]
        for cid in top10_ids if str(cid) in cluster_details
    }

    out_data = {
        "window": cluster_data.get("window", fname.replace("cluster_", "").replace(".json", "")),
        "num_clusters": len(top_details),
        "cluster_details": top_details
    }

    out_name = fname.replace("cluster_", "top10_")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {out_name}")
