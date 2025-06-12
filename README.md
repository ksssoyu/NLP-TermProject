## 📁 Coauthorship Network Analysis Code Documentation (README)

This project performs comprehensive network analysis based on a coauthorship graph. It includes clustering, centrality correlation, power-law fitting, and evaluation of clustering quality.

---

### 🔷 `make_author_json.py`

**Purpose**: Converts the coauthorship graph from GraphML format to JSON.

* Input: `coauthorship_graph_with_year.graphml`
* Output: `coauthorship_graph_with_year.json`
* Why: JSON format is used in the subsequent analysis modules

---

### 🔷 `coauthorship_leiden.py`

**Purpose**: Splits the coauthorship graph into temporal windows and performs clustering using the Leiden algorithm.

* Windows: `2015~2017`, `2016~2018`, ..., `2022~2024`
* Clustering: Uses `leidenalg.find_partition()`
* Includes filtering by minimum cluster size
* Output: `coauthorship_cluster_by_year/cluster_*.json`

---

### 🔷 `top10_cluster_year.py`

**Purpose**: From each time window's clustering result, selects the top 10 clusters with the highest internal collaboration strength.

* Input: `coauthorship_cluster_by_year/`
* Output: `top10_clusters_by_strength/`
* Criteria: Normalized internal edge weight sum divided by node count

---

### 🔷 `top60_author_per_cluster.py`

**Purpose**: For each of the top 10 clusters, ranks and selects the top 60 authors by internal collaboration weight.

* Input: `top10_clusters_by_strength/`
* Output: `top60_authors_per_cluster/`
* Metric: Total internal edge weights

---

### 🔷 `evaluate_cluster_all.py`

**Purpose**: Evaluates the overall quality of clustering on the full graph.

* Metrics: Modularity, Coverage, Conductance
* Input: JSON with global cluster IDs (`coauthorship_graph_with_year_cluster.json`)
* Output: Console summary of scores

---

### 🔷 `evaluate_cluster_per_year.py`

**Purpose**: Evaluates clustering quality for each temporal window.

* Input: `coauthorship_graph_with_year.json`, `coauthorship_cluster_by_year/`
* Metrics: Modularity, Coverage, Performance, Conductance
* Output: DataFrame of results per window

---

### 🔷 `spearman.py`

**Purpose**: Computes correlation between centrality metrics.

* Metrics: Coreness vs. Strength
* Method: Spearman rank correlation (`scipy.stats.spearmanr`)
* Example output: `Spearman ρ(coreness, strength) = 0.959  (p=0)`

---

### 🔷 `power_law.py`

**Purpose**: Analyzes degree distribution of the entire coauthorship network and fits a power-law model.

* Library: `powerlaw`
* Output: Estimated α, xmin, KS distance, log–log CCDF plot
* Interpretation: Quantifies scale-free nature of scientific collaboration networks

---

### 🔧 Environment

* Python >= 3.8
* Required libraries: `networkx`, `igraph`, `leidenalg`, `matplotlib`, `powerlaw`, `scipy`, `pandas`

```bash
pip install networkx igraph leidenalg matplotlib powerlaw scipy pandas
```

---

### 📂 Overall Execution Flow

1. `make_author_json.py`: Convert GraphML → JSON
2. `coauthorship_leiden.py`: Perform time-based clustering
3. `top10_cluster_year.py`: Filter top 10 clusters by strength
4. `top60_author_per_cluster.py`: Select top 60 authors per cluster
5. `evaluate_cluster_per_year.py`: Evaluate each time window
6. `evaluate_cluster_all.py`: Evaluate entire network
7. `power_law.py`: Degree distribution and power-law fitting
8. `spearman.py`: Centrality correlation analysis

---

Refer to each section above for further details. More in-depth documentation with code comments can be added if needed.
