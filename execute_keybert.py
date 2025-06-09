import json
import networkx as nx
import ast
from collections import defaultdict

# 🔧 파일 경로 설정
GRAPH_JSON = "citation_graph_with_cluster_v2.json"    # 입력 파일: 논문 + 클러스터 + citation 정보
OUTPUT_JSON = "cluster_author_pagerank.json"          # 출력 파일: 클러스터별 저자 PageRank 결과
TOP_K_AUTHORS = 5                                     # 클러스터별 추출할 상위 저자 수

# 📥 JSON 파일 로드
with open(GRAPH_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]  # 논문 노드 정보
links = data["links"]  # 논문 간 citation 정보

# 📌 클러스터별 논문 및 저자 정보 수집
cluster_to_papers = defaultdict(list)  # 클러스터 ID → 논문 ID 리스트
paper_authors = {}                     # 논문 ID → 저자 리스트
for node in nodes:
    pid = node["id"]
    group = node["group"]  # 클러스터 ID
    authors = ast.literal_eval(node.get("authors", "[]"))  # 문자열을 실제 리스트로 변환
    cluster_to_papers[group].append(pid)
    paper_authors[pid] = authors

# 📌 클러스터 내부 citation 링크 수집
edges_by_cluster = defaultdict(list)
for link in links:
    src = link["source"]
    tgt = link["target"]
    weight = link.get("weight", 1.0)

    # 각각의 논문이 속한 클러스터 ID 확인
    src_group = next((node["group"] for node in nodes if node["id"] == src), None)
    tgt_group = next((node["group"] for node in nodes if node["id"] == tgt), None)

    # 동일 클러스터 내 citation만 저장
    if src_group == tgt_group and src_group is not None:
        edges_by_cluster[src_group].append((src, tgt, weight))

# 📊 클러스터별 저자 PageRank 저장할 딕셔너리
author_rank_by_cluster = {}

# 🔁 클러스터별로 PageRank 수행
for cluster_id in sorted(cluster_to_papers.keys()):
    paper_ids = set(cluster_to_papers[cluster_id])         # 해당 클러스터의 논문 ID 집합
    edges = edges_by_cluster.get(cluster_id, [])           # 내부 citation 링크

    # 클러스터 내부 논문으로 Directed Graph 생성
    G = nx.DiGraph()
    G.add_nodes_from(paper_ids)
    for src, tgt, w in edges:
        G.add_edge(src, tgt, weight=w)

    # 빈 그래프는 분석하지 않음
    if len(G) == 0:
        continue

    # 📌 논문 단위 PageRank 계산
    pr = nx.pagerank(G, weight="weight")

    # 📌 논문 PageRank를 저자에게 누적
    author_scores = defaultdict(float)
    for pid, score in pr.items():
        for author in paper_authors.get(pid, []):
            author_scores[author] += score  # 다저자 논문은 모두에게 동일 점수 부여

    # 🔄 전체 누적 점수를 1로 정규화 (상대적 비교 가능하게 함)
    total_score = sum(author_scores.values())
    if total_score > 0:
        for author in author_scores:
            author_scores[author] /= total_score

    # 🔝 상위 저자 K명 추출 (PageRank 기준 정렬)
    top_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_AUTHORS]

    # 결과 저장: { cluster_id: [{author: str, score: float}, ...] }
    author_rank_by_cluster[cluster_id] = [
        {"author": author, "score": score} for author, score in top_authors
    ]

# 📤 전체 결과 JSON 파일로 저장
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(author_rank_by_cluster, f, ensure_ascii=False, indent=2)

print(f"✅ 클러스터별 저자 PageRank 저장 완료: {OUTPUT_JSON}")
