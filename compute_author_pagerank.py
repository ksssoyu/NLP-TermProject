# =============================================
# File: compute_author_pagerank.py
# Description:
#   이 스크립트는 coauthorship_graph_with_year_cluster.json 파일을 기반으로
#   공저자 네트워크를 구성하고, 각 저자에 대한 PageRank 점수를 계산합니다.
#   계산된 PageRank 점수와 각 저자의 클러스터 ID를 함께
#   top_authors_by_pagerank.json 파일로 저장합니다.
# =============================================

import json
import networkx as nx
import pandas as pd

# 파일 경로
input_file = 'coauthorship_graph_with_year_cluster.json'
output_file = 'top_authors_by_pagerank.json'

# 데이터 로딩
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = {node['id']: node for node in data['nodes']}
links = data['links']

# 무방향 그래프 생성 (협업은 양방향 관계)
G = nx.Graph()

# 노드 추가
for node_id, node_data in nodes.items():
    G.add_node(node_id, cluster_id=node_data.get('cluster_id'))

# 간선 추가
for link in links:
    src = link['source']
    tgt = link['target']
    weight = float(link.get('weight', 1.0))
    G.add_edge(src, tgt, weight=weight)

# PageRank 계산 (가중치 사용)
pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight')


result = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

# 결과 구성
top_authors = []
for author_id, score in result:
    cluster_id = G.nodes[author_id].get('cluster_id')
    top_authors.append({
        'author': author_id,
        'pagerank': score,
        'cluster_id': cluster_id
    })

# 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(top_authors, f, ensure_ascii=False, indent=2)

print(f'✅ 저자 pagerank 저장 완료 → {output_file}')
