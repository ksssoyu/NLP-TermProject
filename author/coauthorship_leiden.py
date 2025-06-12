import json
import os
import igraph as ig
import leidenalg
from collections import defaultdict

# 설정
INPUT_JSON = "coauthorship_graph_with_year.json"
OUTPUT_DIR = "coauthorship_cluster_by_year"
WINDOW_SIZE = 3
STRIDE = 1
START_YEAR = 2015
END_YEAR = 2022
RESOLUTION = 0.8  # 고정 resolution
MIN_SIZE_BY_WINDOW = {
    "2015_2017": 3,
    "2016_2018": 5,
    "2017_2019": 10,
    "2018_2020": 10,
    "2019_2021": 15,
    "2020_2022": 15,
    "2021_2023": 15,
    "2022_2024": 15
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

all_nodes = data["nodes"]
all_links = data["links"]
id_to_index = {node["id"]: idx for idx, node in enumerate(all_nodes)}

summary = []

for window_start in range(START_YEAR, END_YEAR + 1, STRIDE):
    window_end = window_start + WINDOW_SIZE - 1
    window_key = f"{window_start}_{window_end}"
    min_cluster_size = MIN_SIZE_BY_WINDOW.get(window_key, 5)

    print(f"\n🔎 {window_key} 처리 중")

    # 해당 윈도우에 포함된 엣지 추출
    edges = []
    weights = []
    nodes_in_window = set()

    for link in all_links:
        try:
            years = set(map(int, link.get("years", "").split(",")))
        except:
            continue
        if any(window_start <= y <= window_end for y in years):
            src, tgt = link["source"], link["target"]
            if src in id_to_index and tgt in id_to_index:
                edges.append((src, tgt))
                weights.append(link.get("weight", 1.0))
                nodes_in_window.add(src)
                nodes_in_window.add(tgt)

    if not edges:
        print("⚠️ 엣지 없음 → 스킵")
        continue

    sub_nodes = [node for node in all_nodes if node["id"] in nodes_in_window]
    index_map = {node["id"]: idx for idx, node in enumerate(sub_nodes)}

    g = ig.Graph(n=len(sub_nodes), edges=[(index_map[e[0]], index_map[e[1]]) for e in edges], directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=RESOLUTION
    )

    raw_cluster_details = defaultdict(list)
    for idx, cluster_id in enumerate(partition.membership):
        node = sub_nodes[idx]
        raw_cluster_details[cluster_id].append({"id": node["id"]})

    filtered_cluster_details = {
        cid: papers for cid, papers in raw_cluster_details.items()
        if len(papers) >= min_cluster_size
    }

    output = {
        "window": window_key,
        "num_clusters": len(filtered_cluster_details),
        "cluster_details": filtered_cluster_details
    }

    out_path = os.path.join(OUTPUT_DIR, f"cluster_{window_key}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {out_path} (노드 {len(sub_nodes)}개, 클러스터 {len(filtered_cluster_details)}개)")
    summary.append((window_key, len(filtered_cluster_details)))

print("\n📈 클러스터 개수 요약:")
for window, num_clusters in summary:
    print(f"  - {window}: {num_clusters}개")
