"""
cluster_evaluator.py

Load a pickled NetworkX graph and compute internal clustering‐quality metrics
based on a node attribute that holds cluster IDs.

Metrics computed:
  • Silhouette Score           (–1…+1; higher=better)
  • Davies–Bouldin Index       (lower=better)
  • Calinski–Harabasz Index    (higher=better)
  • Modularity                 (–1…+1; higher=better)
"""
import pickle
import argparse
import networkx as nx
import numpy as np
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def load_graph(path: str) -> nx.Graph:
    ext = path.lower().rsplit('.', 1)[-1]
    if ext == 'gpickle':
        with open (path, "rb") as f:
            graph = pickle.load(f)
        return graph
    elif ext == 'graphml':
        return nx.read_graphml(path)
    else:
        raise ValueError(f"Unsupported file extension '.{ext}'. Use .gpickle or .graphml")


def extract_labels(G: nx.Graph, attr_name: str) -> dict:
    """
    Pulls out a dict { node: cluster_id } from G.nodes[*][attr_name].
    Raises KeyError if any node is missing the attribute.
    """
    labels = {}
    for node, data in G.nodes(data=True):
        if attr_name not in data:
            raise KeyError(f"Node {node!r} is missing attribute '{attr_name}'")
        labels[node] = data[attr_name]
    return labels


def evaluate_clustering(G: nx.Graph, labels: dict) -> dict:
    """
    Given a graph and a node->cluster_id mapping, compute:
      - silhouette_score
      - davies_bouldin_score
      - calinski_harabasz_score
      - modularity
    """
    # 1) Fix node order
    nodes = list(G.nodes())

    # 2) Build label list aligned to nodes[]
    y_true = [labels[n] for n in nodes]

    # 3) Build adjacency‐matrix embedding
    A = nx.to_numpy_array(G, nodelist=nodes)

    # 4) Compute scores
    scores = {}
    scores['silhouette']       = silhouette_score(A, y_true, metric='euclidean')
    scores['davies_bouldin']   = davies_bouldin_score(A, y_true)
    scores['calinski_harabasz'] = calinski_harabasz_score(A, y_true)

    # 5) Compute modularity
    communities = {}
    for node, cid in zip(nodes, y_true):
        communities.setdefault(cid, set()).add(node)
    comm_list = list(communities.values())
    scores['modularity'] = modularity(G, comm_list)

    return scores


def main():
    p = argparse.ArgumentParser(
        description="Evaluate clustering quality of a pickled NetworkX graph."
    )
    p.add_argument(
        'graph_path',
        help="Path to .gpickle or .graphml file containing your clustered graph"
    )
    p.add_argument(
        '--attr',
        default='louvain_cluster_id',
        help="Node attribute name for cluster labels (default: 'cluster_id')"
    )
    args = p.parse_args()

    # Load graph
    G = load_graph(args.graph_path)

    # Extract labels
    labels = extract_labels(G, args.attr)

    # Evaluate
    scores = evaluate_clustering(G, labels)

    # Print results
    print("\nClustering quality metrics:\n")
    for name, val in scores.items():
        print(f"{name:20s}: {val:.6f}")
    print()


if __name__ == '__main__':
    main()
