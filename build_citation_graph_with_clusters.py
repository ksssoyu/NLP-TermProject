# ===============================================================
# File: build_citation_graph_with_clusters.py
# Description:
#   This script constructs a clustered citation graph from an input 
#   GraphML file and paper metadata CSV (with Specter embeddings).
#
#   Main Steps:
#   1. Load citation graph and paper embeddings
#   2. Apply Leiden clustering with configurable resolution
#   3. Merge small clusters based on embedding centroid similarity
#   4. Annotate edges with embedding-based cosine similarity
#   5. Export graph as JSON with node and edge attributes
#
#   Output: citation_graph_with_cluster_v3.json
#
#   Notes:
#   - Uses 768-dim Specter embeddings
#   - Cluster merging helps stabilize topic coherence
#   - Edge similarity is used later for semantic citation analysis
# ===============================================================


import networkx as nx
import igraph as ig
import leidenalg as la
import json
import logging
import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

# 파일 경로
INPUT_GRAPH_FILE = "v2_graph.graphml"
PAPER_METADATA_FILE = "v2_papers.csv"
OUTPUT_JSON_FILE = "citation_graph_with_cluster_v3.json"
RESOLUTION_PARAMETER = 0.75
EMBEDDING_DIM = 768

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_paper_embeddings(filepath):
    df = pd.read_csv(filepath)

    def parse_embedding(emb_str):
        if pd.isna(emb_str):
            return np.zeros(EMBEDDING_DIM)
        try:
            return np.array(ast.literal_eval(emb_str))
        except:
            return np.zeros(EMBEDDING_DIM)
        
    df['embedding_vec'] = df['embedding'].apply(parse_embedding)
    embedding_dict = {row['paperId']: row['embedding_vec'] for _, row in df.iterrows()}
    return embedding_dict

def run_leiden(graph_path):
    logging.info(f"Loading graph from {graph_path}...")
    G_nx = nx.read_graphml(graph_path)

    if not G_nx.nodes():
        logging.warning("Graph is empty.")
        return None, None

    logging.info(f"Loaded {G_nx.number_of_nodes()} nodes and {G_nx.number_of_edges()} edges")

    edges = [(u, v, d.get('weight', 1.0)) for u, v, d in G_nx.edges(data=True)]
    G_ig = ig.Graph.TupleList([(e[0], e[1]) for e in edges], directed=True, vertex_name_attr="name")
    G_ig.es['weight'] = [e[2] for e in edges]

    logging.info("Running Leiden...")
    partition = la.find_partition(
        G_ig, 
        la.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=RESOLUTION_PARAMETER,
        seed=42
    )
    logging.info(f"Leiden finished with {len(partition)} clusters")

    node_to_cluster = {G_ig.vs[i]['name']: cluster_id for i, cluster_id in enumerate(partition.membership)}

    return G_nx, node_to_cluster


def apply_embedding_similarity_to_edges(G_nx, embedding_dict):
    logging.info("Calculating embedding similarities for edges (without changing weights)...")
    for u, v, data in G_nx.edges(data=True):
        emb_u = embedding_dict.get(u)
        emb_v = embedding_dict.get(v)
        if emb_u is not None and emb_v is not None:
            sim = cosine_similarity([emb_u], [emb_v])[0, 0]
            sim = max(sim, 0.0)  # 안정성 확보
            data['similarity'] = sim
        else:
            data['similarity'] = 0.0
    logging.info("Similarity annotation completed.")

def export_to_json(G_nx, node_to_cluster, embedding_dict, output_path):
    node_list = []
    for node_id, data in G_nx.nodes(data=True):
        node_list.append({
            "id": node_id,
            "name": data.get("title", ""),
            "year": int(data.get("year", 0)),
            "authors": data.get("authors", ""),
            "citationCount": int(data.get("citationCount", 0)),
            "group": node_to_cluster.get(node_id, -1),
            "embedding": embedding_dict.get(node_id, []).tolist()  # ✅ 추가
        })

    link_list = []
    for source, target, data in G_nx.edges(data=True):
        link_list.append({
            "source": source,
            "target": target,
            "weight": data.get("weight", 1.0),
            "similarity": data.get("similarity", 0.0),
            "isInfluential": data.get("isInfluential", False),
            "intents": data.get("intents", "")
        })

    graph_data = {"nodes": node_list, "links": link_list}

    with open(output_path, "w") as f:
        json.dump(graph_data, f)

    logging.info(f"Exported to {output_path}")

def merge_small_clusters(G_nx, node_to_cluster, embedding_dict, min_size=5):
    logging.info(f"Merging clusters with size < {min_size}...")

    from collections import defaultdict
    cluster_to_nodes = defaultdict(list)
    for node_id, cluster_id in node_to_cluster.items():
        cluster_to_nodes[cluster_id].append(node_id)

    cluster_centroids = {}
    for cluster_id, nodes in cluster_to_nodes.items():
        vectors = [embedding_dict[n] for n in nodes if n in embedding_dict]
        if vectors:
            centroid = np.mean(vectors, axis=0)
        else:
            centroid = np.zeros(EMBEDDING_DIM)
        cluster_centroids[cluster_id] = centroid

    updated_node_to_cluster = node_to_cluster.copy()
    large_clusters = {cid for cid, nodes in cluster_to_nodes.items() if len(nodes) >= min_size}
    small_clusters = {cid for cid, nodes in cluster_to_nodes.items() if len(nodes) < min_size}

    for small_cid in small_clusters:
        small_centroid = cluster_centroids[small_cid]

        best_cid = None
        best_sim = -1
        for large_cid in large_clusters:
            sim = cosine_similarity([small_centroid], [cluster_centroids[large_cid]])[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_cid = large_cid

        if best_cid is not None:
            for node_id in cluster_to_nodes[small_cid]:
                updated_node_to_cluster[node_id] = best_cid
            logging.info(f"Cluster {small_cid} (size={len(cluster_to_nodes[small_cid])}) merged into {best_cid} (sim={best_sim:.4f})")

    return updated_node_to_cluster

def run_full_pipeline(graph_path, metadata_path, output_path):
    embedding_dict = load_paper_embeddings(metadata_path)
    G_nx, node_to_cluster = run_leiden(graph_path)

    node_to_cluster = merge_small_clusters(G_nx, node_to_cluster, embedding_dict, min_size=5)

    apply_embedding_similarity_to_edges(G_nx, embedding_dict)
    export_to_json(G_nx, node_to_cluster, embedding_dict, output_path)

if __name__ == "__main__":
    run_full_pipeline(INPUT_GRAPH_FILE, PAPER_METADATA_FILE, OUTPUT_JSON_FILE)
