import networkx as nx
from networkx.algorithms.community.quality import modularity, partition_quality

def compute_cut_size(G, C):
    """
    Count edges with exactly one endpoint in C.
    """
    return sum(
        1 
        for u, v in G.edges()
        if (u in C) ^ (v in C)   # xor: one in, one out
    )

def cluster_conductance(G, C):
    """
    Conductance(C) = cut_size(C) / vol(C),
    where vol(C) = sum of degrees of nodes in C.
    """
    cut = compute_cut_size(G, C)
    vol  = sum(d for _, d in G.degree(C))
    return cut / vol if vol > 0 else 0.0

from networkx.algorithms.community.quality import modularity, partition_quality

def evaluate_graph(G, attr='louvain_cluster_id'):
    # 1) extract communities
    labels = nx.get_node_attributes(G, attr)
    missing = set(G) - set(labels)
    if missing:
        raise KeyError(f"Missing {attr} on nodes: {missing!r}")

    communities = {}
    for n, cid in labels.items():
        communities.setdefault(cid, set()).add(n)
    comm_list = list(communities.values())

    # 2) coverage & performance (partition_quality returns exactly 2 floats)
    cov, perf = partition_quality(G, comm_list)   # <— only 2 values :contentReference[oaicite:0]{index=0}

    # 3) modularity (compute separately)
    mod = modularity(G, comm_list)

    # 4) average conductance as before
    conds = [cluster_conductance(G, C) for C in comm_list]
    avg_cond = sum(conds) / len(conds)

    return {
        'modularity'      : mod,
        'coverage'        : cov,
        'performance'     : perf,
        'avg_conductance' : avg_cond
    }


# — example usage —
if __name__ == "__main__":
    import pickle
    G = pickle.load(open("nlp_papers_with_cid.gpickle","rb"))
    scores = evaluate_graph(G, attr='louvain_cluster_id')
    for name, val in scores.items():
        print(f"{name:20s}: {val:.6f}")
