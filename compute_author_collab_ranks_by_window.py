# ===============================================================
# File: compute_author_collab_ranks_by_window.py
# Description:
#   This script analyzes the co-authorship graph over sliding 
#   3-year time windows (e.g., 2015–2017, ..., 2022–2024).
#   For each window, it:
#     - Filters collaboration edges active in the time window
#     - Computes the total collaboration weight per author
#     - Produces global author rankings by total collaboration weight
#     - Produces cluster-level rankings based on cluster membership
#   Output: JSON files (one per window) in coauthorship_ranks_by_window/
#           containing global and cluster-level author rankings.
# ===============================================================


import json
from collections import defaultdict
from pathlib import Path

# Load the coauthorship graph
with open('coauthorship_graph_with_year_cluster.json', 'r') as f:
    data = json.load(f)

nodes = data['nodes']
edges = data['links']

# Define the time windows
windows = [
    ('2015', '2017'),
    ('2016', '2018'),
    ('2017', '2019'),
    ('2018', '2020'),
    ('2019', '2021'),
    ('2020', '2022'),
    ('2021', '2023'),
    ('2022', '2024')
]

# Prepare author → cluster mapping
author_cluster = {}
for node in nodes:
    author_cluster[node['id']] = node.get('cluster_id')

# Prepare output directory
output_dir = Path('coauthorship_ranks_by_window')
output_dir.mkdir(exist_ok=True)

# Process each window
for start, end in windows:
    start_year = int(start)
    end_year = int(end)
    window_name = f"{start}_{end}"

    # Filter edges for the time window
    edge_in_window = []
    for e in edges:
        # Handle multi-year strings like "2019,2022,2023"
        years = str(e.get('years', '')).split(',')
        if any(start_year <= int(y.strip()) <= end_year for y in years if y.strip().isdigit()):
            edge_in_window.append(e)

    # Calculate total weights per author
    author_weights = defaultdict(float)
    for e in edge_in_window:
        author_weights[e['source']] += e['weight']
        author_weights[e['target']] += e['weight']

    # Global ranking
    global_rank = sorted(
        [
            {
                "author": a,
                "total_weight": w,
                "cluster_id": author_cluster.get(a)
            }
            for a, w in author_weights.items()
        ],
        key=lambda x: -x["total_weight"]
    )

    for i, item in enumerate(global_rank):
        item["rank"] = i + 1

    # Cluster-level ranking
    cluster_groups = defaultdict(list)
    for item in global_rank:
        cid = item["cluster_id"]
        if cid is not None:
            cluster_groups[cid].append(item)

    cluster_rank = {}
    for cid, members in cluster_groups.items():
        sorted_members = sorted(members, key=lambda x: -x["total_weight"])
        cluster_rank[cid] = [
            {
                "author": m["author"],
                "total_weight": m["total_weight"],
                "rank": i + 1
            }
            for i, m in enumerate(sorted_members)
        ]

    # Save to JSON
    output_path = output_dir / f'global_and_cluster_rank_{window_name}.json'
    with open(output_path, 'w') as f:
        json.dump({
            "global_rank": global_rank,
            "cluster_rank": cluster_rank
        }, f, indent=2)

    print(f"✅ Saved: {output_path}")
