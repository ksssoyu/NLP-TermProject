import json, networkx as nx, numpy as np
from scipy.stats import spearmanr

# JSON 로드
with open("coauthorship_graph_with_year_cluster.json", encoding="utf-8") as f:
    d = json.load(f)

G = nx.Graph()
for n in d['nodes']:  # 노드 추가
    G.add_node(n['id'])
for l in d['links']:  # 엣지 추가 (weight 포함)
    G.add_edge(l['source'], l['target'], weight=l.get('weight', 1))
# 0. self-loop 제거 (딱 한 줄)
G.remove_edges_from(nx.selfloop_edges(G))
# coreness, strength 계산
core  = nx.core_number(G)  # {node: coreness}
stren = {v: sum(w['weight'] for _,_,w in G.edges(v, data=True)) for v in G}

# Spearman ρ
common = core.keys()
rho, p = spearmanr([core[v]   for v in common],
                   [stren[v]  for v in common])
print(f"Spearman ρ(coreness, strength) = {rho:.3f}  (p={p:.1g})")
