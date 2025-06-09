import json
import networkx as nx
import ast
from collections import defaultdict

# 🔧 파일 경로
GRAPH_JSON = "citation_graph_with_cluster_v2.json"
OUTPUT_JSON = "cluster_author_pagerank.json"
TOP_K_AUTHORS = 5

# 📥 그래프 JSON 로드
with open(GRAPH_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]
links = data["links"]

# 📌 클러스터별 논문 ID 및 저자 정보 수집
cluster_to_papers = defaultdict(list)
paper_authors = {}
for node in nodes:
    pid = node["id"]
    group = node["group"]
    authors = ast.literal_eval(node.get("authors", "[]"))  # authors 문자열을 리스트로 변환
    cluster_to_papers[group].append(pid)
    paper_authors[pid] = authors

# 📌 클러스터 내부 citation link 모으기
edges_by_cluster = defaultdict(list)
for link in links:
    src = link["source"]
    tgt = link["target"]
    weight = link.get("weight", 1.0)

    src_group = next((node["group"] for node in nodes if node["id"] == src), None)
    tgt_group = next((node["group"] for node in nodes if node["id"] == tgt), None)

    if src_group == tgt_group and src_group is not None:
        edges_by_cluster[src_group].append((src, tgt, weight))

# 📊 클러스터별 저자 PageRank 결과 저장
author_rank_by_cluster = {}

for cluster_id in sorted(cluster_to_papers.keys()):
    paper_ids = set(cluster_to_papers[cluster_id])
    edges = edges_by_cluster.get(cluster_id, [])

    G = nx.DiGraph()
    G.add_nodes_from(paper_ids)
    for src, tgt, w in edges:
        G.add_edge(src, tgt, weight=w)

    if len(G) == 0:
        continue

    pr = nx.pagerank(G, weight="weight")

    author_scores = defaultdict(float)
    for pid, score in pr.items():
        for author in paper_authors.get(pid, []):
            author_scores[author] += score

    # 🔄 전체 누적 점수를 1로 정규화
    total_score = sum(author_scores.values())
    if total_score > 0:
        for author in author_scores:
            author_scores[author] /= total_score

    # 🔝 상위 저자 추출
    top_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_AUTHORS]

    # 저장 형식: { "cluster_id": [{"author": name, "score": value}, ...] }
    author_rank_by_cluster[cluster_id] = [
        {"author": author, "score": score} for author, score in top_authors
    ]

# 📤 JSON 저장
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(author_rank_by_cluster, f, ensure_ascii=False, indent=2)

print(f"✅ 클러스터별 저자 PageRank 저장 완료: {OUTPUT_JSON}")
