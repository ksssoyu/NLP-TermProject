import pickle
import random
import networkx as nx
import numpy as np
from scipy.stats import spearmanr

# ----------------------------
# 1. 평가 함수 정의
# ----------------------------

def stability_score(G, centrality_fn, drop_frac=0.05, trials=10):
    base_scores = centrality_fn(G)
    base_order = [node for node, _ in sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)]
    rhos = []

    m = G.number_of_edges()
    drop = int(m * drop_frac)

    for _ in range(trials):
        H = G.copy()
        removed = random.sample(list(G.edges()), drop)
        H.remove_edges_from(removed)

        scores_h = centrality_fn(H)
        order_h = [node for node, _ in sorted(scores_h.items(), key=lambda kv: kv[1], reverse=True)]
        rho, _ = spearmanr(base_order, order_h)
        rhos.append(rho)

    return np.mean(rhos)

def centralization(scores):
    vals = np.array(list(scores.values()))
    diff = vals.max() - vals
    denom = (len(vals) - 1) * vals.max()
    return diff.sum() / denom if denom > 0 else 0.0

def gini(xs):
    xs = sorted(xs)
    n = len(xs)
    cum = sum((i + 1) * v for i, v in enumerate(xs))
    total = sum(xs)
    return (2 * cum) / (n * total) - (n + 1) / n

def simpson_index(xs):
    xs = np.array(xs)
    return np.sum(xs ** 2)

# ----------------------------
# 2. 그래프 로딩 & 평가
# ----------------------------

with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

# 가장 큰 연결 성분만 추출
largest = max(nx.connected_components(G), key=len)
G_sub = G.subgraph(largest).copy()

# 중심성 계산 함수 지정
def eigen_fn(graph):
    return nx.eigenvector_centrality(graph, weight='weight', max_iter=1000)

# 평가 수행
eig_scores = eigen_fn(G_sub)
eig_vals = list(eig_scores.values())

stab = stability_score(G_sub, eigen_fn, drop_frac=0.05, trials=5)
cent = centralization(eig_scores)
gini_val = gini(eig_vals)
simp = simpson_index(eig_vals)

# 결과 출력
print(f"Eigenvector Stability:       {stab:.3f}")
print(f"Eigenvector Centralization:  {cent:.3f}")
print(f"Eigenvector Gini Coefficient:{gini_val:.3f}")
print(f"Eigenvector Simpson Index:   {simp:.3f}")
