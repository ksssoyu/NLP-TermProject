#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coauthorship_degree_powerlaw.py

▸ 논문 저자 공동-저자 그래프(JSON) → 차수 분포 분석
▸ 로그–로그 분포 시각화 + power-law 지수 α 추정
"""

import json
import collections
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import pathlib
import sys

# --------------------------------------------------
# 0) 설정 ­­­
# --------------------------------------------------
JSON_PATH = "coauthorship_graph_with_year_cluster.json"  # 수정 가능

# --------------------------------------------------
# 1) 그래프 로드
# --------------------------------------------------
p = pathlib.Path(JSON_PATH)
if not p.exists():
    sys.exit(f"❌ 파일을 찾을 수 없습니다: {p.resolve()}")

with p.open(encoding="utf-8") as f:
    data = json.load(f)

G = nx.Graph()
for n in data["nodes"]:
    G.add_node(n["id"])
for l in data["links"]:
    G.add_edge(l["source"], l["target"], weight=l.get("weight", 1.0))

# --------------------------------------------------
# 2) Degree distribution statistics
# --------------------------------------------------
deg_dict = dict(G.degree())
degrees  = list(deg_dict.values())

N_nodes = len(degrees)
avg_deg = sum(degrees) / N_nodes

print("=== Basic Statistics ===")
print(f"Number of nodes    : {N_nodes:,}")
print(f"Number of edges    : {G.number_of_edges():,}")
print(f"Average degree     : {avg_deg:.2f}")
print()

# --------------------------------------------------
# 3) Degree frequency table & log–log plot
# --------------------------------------------------
freq = collections.Counter(degrees)
print("=== Top 10 Degree Frequencies ===")
for k, cnt in freq.most_common(10):
    print(f"Degree {k:>3} : {cnt:>6}")
print()

# --------------------------------------------------
# 4) Power-law fitting results
# --------------------------------------------------
fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
alpha = fit.power_law.alpha
xmin  = fit.power_law.xmin
ks_D  = fit.power_law.D

print("=== Power-law Fit Results ===")
print(f"α (exponent)       : {alpha:.3f}")
print(f"xmin               : {xmin}")
print(f"KS distance        : {ks_D:.4f}")
print()

# CCDF + 적합선
fig = fit.plot_ccdf(linewidth=2, label="empirical")
fit.power_law.plot_ccdf(ax=fig, color="r",
                        linestyle="--",
                        label=f"power-law fit α={alpha:.2f}")
plt.legend()
plt.title("CCDF with power-law fit")
plt.tight_layout()
plt.show()
