import pickle
import random
import networkx as nx
import numpy as np
from scipy.stats import spearmanr

def stability_score(G, centrality_fn, drop_frac=0.05, trials=10):
    """
    Compute average Spearman-rho between the original ranking and
    rankings on G with a random drop_frac of edges removed.
    """
    # 1) baseline scores & ordering
    base_scores = centrality_fn(G)
    base_order = [node for node, _ in sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)]

    rhos = []
    m = G.number_of_edges()
    drop = int(m * drop_frac)

    for _ in range(trials):
        # 2) make a perturbed copy
        H = G.copy()
        removed = random.sample(list(G.edges()), drop)
        H.remove_edges_from(removed)

        # 3) recompute & compare
        scores_h = centrality_fn(H)
        order_h  = [n for n, _ in sorted(scores_h.items(), key=lambda kv: kv[1], reverse=True)]
        rho, _   = spearmanr(base_order, order_h)
        rhos.append(rho)

    return np.mean(rhos)

def centralization(scores: dict):
    vals = np.array(list(scores.values()))
    diff = vals.max() - vals
    # normalize by the sum you’d get if one node had all the weight:
    denom = (len(vals)-1) * vals.max()
    return diff.sum() / denom if denom>0 else 0.0

def gini(xs):
    """Gini coefficient between 0 (perfect equality) and 1 (perfect inequality)."""
    xs = sorted(xs)
    n = len(xs)
    cum = sum((i+1)*v for i,v in enumerate(xs))
    total = sum(xs)
    return (2*cum)/(n*total) - (n+1)/n



def get_largest_subgraph(G):
    """
    Identifies and returns the largest connected component as an undirected subgraph
    from the input graph G.

    Args:
        G (nx.Graph or nx.DiGraph): The input graph.

    Returns:
        nx.Graph or None: The largest connected component as an undirected subgraph.
                         Returns None if the input graph is empty, has no nodes,
                         or if an error occurs during processing.
                         Note: This function *always* returns the largest component,
                         even if it contains fewer than 3 nodes. If you need to
                         filter by size, add a check after calling this function.
    """
    # Handle empty or trivial graphs immediately
    if not G or G.number_of_nodes() == 0:
        return None

    # Convert the graph to undirected to find connected components
    G_undirected = G.to_undirected()

    try:
        # Get all connected components
        # nx.connected_components returns an iterator of sets of nodes
        connected_components = list(nx.connected_components(G_undirected))

        # If for some reason no components are found (e.g., a very unusual graph object)
        if not connected_components:
            return None

        # Find the set of nodes belonging to the largest component
        # The 'key=len' argument makes max() find the set with the most elements
        largest_cc_nodes = max(connected_components, key=len)

        # Create a subgraph from these nodes
        # .copy() is important to ensure it's a new graph object, not just a view
        G_largest = G_undirected.subgraph(largest_cc_nodes).copy()

        # Removed the 'if G_undirected.number_of_nodes() < 3' block
        # and the 'eigenvector_scores = {}' assignment, as the function's
        # sole purpose now is to return the subgraph, not compute centrality,
        # and it should return the largest component regardless of its size.
        # The caller can then decide if the returned subgraph is "too small"
        # for subsequent operations like eigenvector centrality.

        return G_largest

    except Exception as e:
        # Catch any unexpected errors that might occur during graph processing
        print(f"An error occurred while finding the largest subgraph: {e}")
        return None

with open("nlp_papers_with_cid.gpickle", "rb") as f:
    G = pickle.load(f)

pr_stab = stability_score(G, nx.pagerank, drop_frac=0.05, trials=5)
print(f"PageRank stability: {pr_stab:.3f}")
pr_scores = nx.pagerank(G)

largest_subgraph = get_largest_subgraph(G)
pr_scores_subgraph = nx.pagerank(largest_subgraph)

pr_cent = centralization(pr_scores_subgraph)
print(f"PageRank centralization: {pr_cent:.3f}")
pr = nx.pagerank(G)
top10 = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:10]
print(top10)
print("Sum of scores:", sum(pr.values()))
pr_vals = list(pr.values())
print("PageRank Gini:", gini(pr_vals))
simpson = np.sum(np.array(pr_vals)**2)    # closer to 1 = extreme concentration
print("PageRank Simpson index:", simpson)