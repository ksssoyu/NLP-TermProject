"""
Extract Top 60 Authors per Cluster (Per Time Window)

This script processes co-authorship graph data and time-windowed top 10 cluster information 
to extract the top 60 authors (by internal collaboration strength) for each cluster in each window.

Functionality:
- Loads the full co-authorship graph (`coauthorship_graph_with_year.json`)
- Iterates through time-windowed top-10 cluster files (e.g., `top10_clusters_by_strength/`)
- For each window and cluster, filters edges by overlapping years
- Computes internal collaboration weight for authors
- Extracts top 60 authors per cluster based on total internal collaboration weight
- Outputs one JSON file per time window to `top60_authors_per_cluster/`

Inputs:
- `coauthorship_graph_with_year.json`: Full co-authorship graph with yearly edge data
- `top10_clusters_by_strength/*.json`: Cluster membership info for each time window

Output:
- JSON files in `top60_authors_per_cluster/` with the format:
  {
    "window": "2017_2019",
    "top5_authors_per_cluster": {
        "3": [
            {"author": "Author A", "total_internal_collab_weight": 13.0},
            ...
        ],
        ...
    }
  }

Usage:
- Run directly: `python extract_top60_authors_per_cluster.py`
"""

import json
import os
from collections import defaultdict

GRAPH_PATH = "coauthorship_graph_with_year.json"
TOP10_CLUSTER_DIR = "top10_clusters_by_strength"  # ← 경로 맞게 수정
OUTPUT_DIR = "top60_authors_per_cluster"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 전체 링크 불러오기
with open(GRAPH_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
all_links = data["links"]

# 연도별 top10 클러스터 파일 반복
for fname in sorted(os.listdir(TOP10_CLUSTER_DIR)):
    if not fname.endswith(".json"):
        continue

    with open(os.path.join(TOP10_CLUSTER_DIR, fname), "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    window = cluster_data["window"]
    start_year, end_year = map(int, window.split("_"))
    valid_years = set(range(start_year, end_year + 1))

    # 해당 연도에 속한 링크만 필터링
    filtered_links = []
    for link in all_links:
        try:
            years = set(map(int, link.get("years", "").split(",")))
            if years & valid_years:
                filtered_links.append(link)
        except:
            continue

    # 클러스터별 Top 60 author 계산
    cluster_details = cluster_data["cluster_details"]
    result = {}

    for cid, members in cluster_details.items():
        node_set = set(m["id"] for m in members)
        author_score = defaultdict(float)

        for link in filtered_links:
            src, tgt = link["source"], link["target"]
            w = float(link.get("weight", 1.0))
            if src in node_set and tgt in node_set:
                author_score[src] += w
                author_score[tgt] += w  # 무방향

        top5 = sorted(author_score.items(), key=lambda x: x[1], reverse=True)[:60]
        result[cid] = [
            {"author": aid, "total_internal_collab_weight": score}
            for aid, score in top5
        ]

    # 저장
    out_name = fname.replace("top10_", "top60_authors_")
    with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
        json.dump({
            "window": window,
            "top5_authors_per_cluster": result
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {out_name}")
