import networkx as nx
import igraph as ig
import leidenalg as la
import json
import logging
import pandas as pd
import numpy as np
import ast
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# 파일 경로
INPUT_GRAPH_FILE = "v2_graph.graphml"
PAPER_METADATA_FILE = "v2_papers.csv"
OUTPUT_JSON_FILE = "citation_graph_with_cluster_v2.json"
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
    logging.info("Applying embedding similarity to edge weights...")
    for u, v, data in G_nx.edges(data=True):
        emb_u = embedding_dict.get(u)
        emb_v = embedding_dict.get(v)
        if emb_u is not None and emb_v is not None:
            sim = cosine_similarity([emb_u], [emb_v])[0,0]
            sim = max(sim, 0.0)  # 음수 방지 (혹시 몰라 안정성용)
            data['weight'] *= sim
    logging.info("Edge weight update completed.")

def export_to_json(G_nx, node_to_cluster, embedding_dict, output_path):
    node_list = []
    for node_id, data in G_nx.nodes(data=True):
        node_list.append({
            "id": node_id,
            "name": data.get("title", ""),
            "year": int(data.get("year", 0)) if data.get("year") else 0,
            "authors": data.get("authors", ""),
            "citationCount": int(data.get("citationCount", 0)) if data.get("citationCount") else 0,
            "group": node_to_cluster.get(node_id, -1),
            # PCA는 이제 생략
        })

    link_list = []
    for source, target, data in G_nx.edges(data=True):
        emb_u = embedding_dict.get(source, np.zeros(EMBEDDING_DIM))
        emb_v = embedding_dict.get(target, np.zeros(EMBEDDING_DIM))
        sim = cosine_similarity([emb_u], [emb_v])[0,0]
        sim = max(sim, 0.0)

        link_list.append({
            "source": source,
            "target": target,
            "weight": data.get("weight", 1.0),
            "similarity": sim,    # 추가!
            "isInfluential": data.get("isInfluential", False),
            "intents": data.get("intents", "")
        })

    graph_data = { "nodes": node_list, "links": link_list }

    with open(output_path, "w") as f:
        json.dump(graph_data, f)

    logging.info(f"Exported to {output_path}")


def run_full_pipeline(graph_path, metadata_path, output_path):
    embedding_dict = load_paper_embeddings(metadata_path)
    G_nx, node_to_cluster = run_leiden(graph_path)
    apply_embedding_similarity_to_edges(G_nx, embedding_dict)
    export_to_json(G_nx, node_to_cluster, embedding_dict, output_path)

if __name__ == "__main__":
    run_full_pipeline(INPUT_GRAPH_FILE, PAPER_METADATA_FILE, OUTPUT_JSON_FILE)
