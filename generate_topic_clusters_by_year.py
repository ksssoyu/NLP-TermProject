"""
Generate Topic Clusters from Citation Graph for Sliding Time Windows (Leiden + Custom Weights)

This script performs temporal clustering of papers in a citation graph across multiple 
3-year sliding windows using the Leiden community detection algorithm.

📌 Key Features:
- Filters nodes (papers) and edges within each time window (e.g., 2015–2017, ..., 2022–2024)
- Applies a custom edge weighting strategy:
    - Citation intent-based weights: background (0.5), methodology (1.0), result (0.8)
    - Additional 1.0 weight for influential citations
- Uses `leidenalg` over an undirected igraph to find paper clusters per window
- Dynamically adjusts the resolution parameter to target 10–15 clusters (adaptive)
- Filters out small clusters below a specified threshold (per window)
- Saves each result as a JSON file with cluster details (paper IDs and titles)

📁 Input:
- `citation_graph_with_cluster_v3.json` containing:
  - nodes: [{id, title/name, year, ...}]
  - links: [{source, target, intents, isInfluential, ...}]

📁 Output:
- Folder: `cluster_results_by_year/`
- Files: `cluster_<window>.json` per time window (e.g., cluster_2017_2019.json)
- Each file includes:
  {
    "window": "2017_2019",
    "num_clusters": 12,
    "cluster_details": {
        "0": [{"id": "paper1", "title": "Transformer Models"}, ...],
        ...
    }
  }

⚙️ Settings:
- WINDOW_SIZE = 3 (years), STRIDE = 1
- Resolution tuning loop until 10–15 clusters are found
- Per-window minimum cluster sizes (MIN_SIZE_BY_WINDOW)
- Custom citation weight mapping via intent types

Usage:
- Run with: `python generate_topic_clusters_by_year.py`
"""


import json
import os
import igraph as ig
import leidenalg
from collections import defaultdict

# 설정
INPUT_JSON = "citation_graph_with_cluster_v3.json"
OUTPUT_DIR = "cluster_results_by_year"
WINDOW_SIZE = 3
STRIDE = 1
START_YEAR = 2015
END_YEAR = 2022
MIN_SIZE_BY_WINDOW = {
    "2015_2017": 3,
    "2016_2018": 5,
    "2017_2019": 15,
    "2018_2020": 15,
    "2019_2021": 30,
    "2020_2022": 30,
    "2021_2023": 30,
    "2022_2024": 30
}
TARGET_MIN = 10
TARGET_MAX = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

all_nodes = data["nodes"]
all_links = data["links"]

INTENT_WEIGHT = {
    "background": 0.5,
    "methodology": 1.0,
    "result": 0.8
}

def compute_weight(link):
    base_weight = 0.0
    intents_str = link.get("intents", "")
    if intents_str:
        for intent in intents_str.split(","):
            base_weight += INTENT_WEIGHT.get(intent.strip(), 0.0)
    if link.get("isInfluential", False):
        base_weight += 1.0
    return max(base_weight, 0.0)

summary = []

for window_start in range(START_YEAR, END_YEAR + 1, STRIDE):
    window_end = window_start + WINDOW_SIZE - 1
    window_key = f"{window_start}_{window_end}"
    min_cluster_size = MIN_SIZE_BY_WINDOW.get(window_key, 5)

    print(f"\n🔎 {window_key} 처리 중")
    filtered_nodes = [n for n in all_nodes if window_start <= n.get("year", 0) <= window_end]
    id_to_index = {n["id"]: idx for idx, n in enumerate(filtered_nodes)}
    valid_ids = set(id_to_index.keys())

    edges = []
    weights = []

    for link in all_links:
        if link["source"] in valid_ids and link["target"] in valid_ids:
            src = id_to_index[link["source"]]
            tgt = id_to_index[link["target"]]
            w = compute_weight(link)
            if w >= 1.0:
                edges.append((src, tgt))
                weights.append(w)

    if not edges:
        print("⚠️ 엣지 없음 → 스킵")
        continue

    g = ig.Graph(n=len(filtered_nodes), edges=edges, directed=False)
    g.es["weight"] = weights

    resolution = 0.4
    step = 0.1
    max_trials = 50
    filtered_cluster_details = {}

    for _ in range(max_trials):
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution
        )

        raw_cluster_details = defaultdict(list)
        for idx, cluster_id in enumerate(partition.membership):
            node = filtered_nodes[idx]
            raw_cluster_details[cluster_id].append({
                "id": node["id"],
                "title": node.get("title") or node.get("name", "")
            })

        filtered_cluster_details = {
            cid: papers for cid, papers in raw_cluster_details.items()
            if len(papers) >= min_cluster_size
        }

        num_clusters = len(filtered_cluster_details)
        print(f"  🔎 resolution={resolution:.2f} → {num_clusters}개 클러스터")

        if TARGET_MIN <= num_clusters <= TARGET_MAX:
            break

        resolution += step

    output = {
        "window": window_key,
        "num_clusters": len(filtered_cluster_details),
        "cluster_details": filtered_cluster_details
    }

    out_path = os.path.join(OUTPUT_DIR, f"cluster_{window_key}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {out_path} (논문 {len(filtered_nodes)}개, 클러스터 {len(filtered_cluster_details)}개)")
    summary.append((window_key, len(filtered_cluster_details)))

print("\n📈 클러스터 개수 요약:")
for window, num_clusters in summary:
    print(f"  - {window}: {num_clusters}개")
