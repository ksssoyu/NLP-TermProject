import json
import networkx as nx
from collections import defaultdict

# 🔧 파일 설정
INPUT_JSON = "citation_graph_with_cluster_v2.json"   # 논문+클러스터+citation 정보를 담은 입력 파일
OUTPUT_JSON = "top_clusters_for_keyword.json"        # 출력 파일 (Top K 클러스터 및 논문 목록 저장)
TOP_K = 5                                             # PageRank 기준 상위 K개 클러스터 추출

# 📥 JSON 로드
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]  # 논문 리스트
links = data["links"]  # citation 관계 리스트

# 📌 논문 ID → 클러스터 ID 매핑
paper_to_cluster = {node["id"]: node["group"] for node in nodes}

# 📌 클러스터 간 citation 가중치 누적
# 논문 단위 citation을 클러스터 단위로 치환
cluster_edges = defaultdict(float)
for link in links:
    src, tgt = link["source"], link["target"]
    w = link.get("weight", 1.0)
    
    c1 = paper_to_cluster.get(src)  # source 논문이 속한 클러스터
    c2 = paper_to_cluster.get(tgt)  # target 논문이 속한 클러스터
    
    if c1 is not None and c2 is not None and c1 != c2:
        cluster_edges[(c1, c2)] += w  # 같은 클러스터 내는 제외, 클러스터 간만 누적

# 📌 클러스터 간 Directed Graph 구성
G_cluster = nx.DiGraph()
for (c1, c2), w in cluster_edges.items():
    G_cluster.add_edge(c1, c2, weight=w)

# 📌 PageRank 수행: 클러스터 간 영향력 측정
pagerank_scores = nx.pagerank(G_cluster, weight="weight")

# 📌 PageRank 기준 상위 K개 클러스터 추출
top_clusters = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
top_cluster_ids = [cid for cid, _ in top_clusters]

# 📌 클러스터별 논문 정보 모으기 (ID + 제목)
cluster_details = defaultdict(list)
for node in nodes:
    cid = node["group"]
    if cid in top_cluster_ids:
        cluster_details[cid].append({
            "id": node["id"],
            "title": node["name"]
        })

# 📦 최종 저장할 데이터 구성
output_data = {
    "top_clusters": top_cluster_ids,       # 상위 K개 클러스터 ID 리스트
    "cluster_details": cluster_details,    # 각 클러스터에 속한 논문 ID + 제목
    "cluster_pagerank": pagerank_scores    # 전체 클러스터의 PageRank 점수 (비교용)
}

# 📤 JSON 파일로 저장
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ 저장 완료: {OUTPUT_JSON}")
