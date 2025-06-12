import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
import json
import logging
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# The path to the graph file.
INPUT_GRAPH_FILE = "final_v2_graph.graphml"
# The path to the new CSV file that includes the embedding vectors.
INPUT_EMBEDDING_CSV = "papers_with_embeddings.csv"
# Directory to save the output visualizations
OUTPUT_DIR = "combined_cluster_visualizations"

# --- Analysis Configuration ---
# Number of top communities/topics to display in the final report.
TOP_N_COMMUNITIES = 5
# Number of top papers to show from each community/topic.
PAPERS_PER_GROUP = 8

# --- Leiden Algorithm Configuration ---
LEIDEN_RESOLUTION = 0.88

# --- K-Means Clustering Configuration ---
# The number of topical clusters to find. This is a key parameter to tune.
# A good starting point is often between 20 and 50 for a dataset of this size.
NUM_TOPICAL_CLUSTERS = 10

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Functions ---

def visualize_community_topics(subgraph, community_id, df_community):
    """
    Creates and saves a visualization of a single community, with nodes
    colored by their topical cluster.
    """
    if not subgraph.nodes():
        logging.warning(f"Skipping visualization for {community_id} as its subgraph is empty.")
        return

    logging.info(f"Generating visualization for {community_id}...")
    
    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
    
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
    
    # Node colors based on Topical Cluster ID
    # Map topic IDs (e.g., "Topic_5") to integers for coloring
    unique_topics = list(df_community['topical_cluster'].dropna().unique())
    topic_to_color_id = {topic: i for i, topic in enumerate(unique_topics)}
    node_colors = [topic_to_color_id.get(df_community.loc[n].get('topical_cluster'), -1) for n in subgraph.nodes()]
    
    cmap = plt.cm.get_cmap('viridis', len(unique_topics))
    
    # Node sizes based on PageRank
    min_size, max_size = 20, 2000
    pr_values = [df_community.loc[n].get('pagerank', 0) for n in subgraph.nodes()]
    max_pr = max(pr_values) if pr_values else 0
    if max_pr > 0:
        scaled_sizes = [min_size + (max_size - min_size) * (v / max_pr) for v in pr_values]
    else:
        scaled_sizes = [min_size] * len(pr_values)
        
    # Draw Graph
    nx.draw_networkx_nodes(subgraph, pos, node_size=scaled_sizes, node_color=node_colors, cmap=cmap, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, width=0.5, edge_color='grey', alpha=0.5, arrows=False, ax=ax)
    
    # Finalize and Save
    ax.set_title(f"Thematic Makeup of {community_id}", fontdict={'fontsize': 24})
    ax.axis('off')
    
    try:
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, f"community_{community_id.replace('Community_', '')}_topics.png")
        plt.savefig(output_path, bbox_inches='tight')
        logging.info(f"Visualization for {community_id} saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save visualization file for {community_id}: {e}")
        
    plt.close(fig)

def run_leiden_clustering(G):
    """Performs Leiden clustering on a networkx graph."""
    logging.info("Starting Leiden clustering (Network Structure)...")
    if not G.nodes() or not G.edges():
        logging.warning("Graph has no nodes or edges for Leiden clustering.")
        return {}

    ordered_nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(ordered_nodes)}
    
    num_vertices = len(ordered_nodes)
    G_ig = ig.Graph(n=num_vertices, directed=True)
    
    edges_with_weights = [(node_to_idx[u], node_to_idx[v], d.get('weight', 1.0)) for u, v, d in G.edges(data=True)]
    if edges_with_weights:
        G_ig.add_edges([e[:2] for e in edges_with_weights])
        G_ig.es['weight'] = [e[2] for e in edges_with_weights]

    partition = la.find_partition(
        G_ig, la.RBConfigurationVertexPartition,
        weights='weight', resolution_parameter=LEIDEN_RESOLUTION, seed=42
    )
    
    leiden_communities = {ordered_nodes[i]: f"Community_{cluster_id}" for i, cluster_id in enumerate(partition.membership)}
    logging.info(f"Found {len(set(partition.membership))} Leiden communities.")
    return leiden_communities

def run_embedding_clustering(df):
    """Performs K-Means clustering on embedding vectors."""
    logging.info("Starting K-Means clustering (Semantic Content)...")
    
    df_with_embeddings = df.dropna(subset=['embedding']).copy()
    if df_with_embeddings.empty:
        logging.warning("No embeddings found in the dataset. Cannot perform K-Means clustering.")
        return {}
        
    logging.info(f"Found {len(df_with_embeddings)} papers with embeddings to cluster.")
    
    embeddings = np.array([json.loads(vec) for vec in df_with_embeddings['embedding']])
    embeddings_normalized = normalize(embeddings)
    
    kmeans = KMeans(n_clusters=NUM_TOPICAL_CLUSTERS, random_state=42, n_init='auto')
    df_with_embeddings['topical_cluster'] = kmeans.fit_predict(embeddings_normalized)
    
    topical_clusters = {row['paperId']: f"Topic_{row['topical_cluster']}" for _, row in df_with_embeddings.iterrows()}
    
    logging.info(f"Found {NUM_TOPICAL_CLUSTERS} topical clusters.")
    return topical_clusters


def combined_analysis(graph_path, csv_path):
    """
    Loads data, performs both network and content clustering, and synthesizes the results.
    """
    # --- 1. Load Data ---
    logging.info(f"Loading graph from {graph_path}...")
    try:
        G = nx.read_graphml(graph_path)
    except FileNotFoundError:
        logging.error(f"File not found: {graph_path}. Cannot proceed.")
        return

    logging.info(f"Loading dataset with embeddings from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        # Using paperId as a column for easier filtering, not index
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}. Make sure the embedding fetcher script has run.")
        return

    # --- 2. Run Both Clustering Algorithms ---
    leiden_communities = run_leiden_clustering(G)
    topical_clusters = run_embedding_clustering(df)
    
    # --- 3. Synthesize Results ---
    df['leiden_community'] = df['paperId'].map(leiden_communities)
    df['topical_cluster'] = df['paperId'].map(topical_clusters)
    
    pagerank_scores = nx.pagerank(G, weight='weight')
    df['pagerank'] = df['paperId'].map(pagerank_scores)
    
    # --- 4. Generate Report and Visualizations ---
    print("\n" + "="*80)
    print("Combined Analysis Report: Leiden Communities and their Topics")
    print("="*80)
    
    top_leiden_communities = df['leiden_community'].value_counts().nlargest(TOP_N_COMMUNITIES).index

    for community_id in top_leiden_communities:
        community_df = df[df['leiden_community'] == community_id]
        print(f"\n--- {community_id} ({len(community_df)} papers) ---")
        
        top_papers_in_community = community_df.nlargest(PAPERS_PER_GROUP, 'pagerank')
        print(f"  Most Influential Papers in this Community (by PageRank):")
        for _, paper in top_papers_in_community.iterrows():
            title = str(paper.get('title', 'N/A'))[:70]
            print(f"    - ({int(paper.get('year', 0))}) {title}")
            
        topic_distribution = community_df['topical_cluster'].value_counts().nlargest(PAPERS_PER_GROUP)
        print(f"\n  Main Topics (by Embedding Similarity) in this Community:")
        if topic_distribution.empty:
            print("    - No topical clusters found for papers in this community.")
        else:
            for topic_id, count in topic_distribution.items():
                if pd.isna(topic_id): continue
                percentage = (count / len(community_df)) * 100
                rep_paper = df[df['topical_cluster'] == topic_id].nlargest(1, 'pagerank').iloc[0]
                rep_title = str(rep_paper.get('title', 'N/A'))[:60]
                print(f"    - {topic_id} ({count} papers, {percentage:.1f}%): e.g., '{rep_title}'")
        
        # --- Call visualization for this community ---
        community_subgraph = G.subgraph(community_df['paperId'].tolist())
        # Pass the main dataframe which now includes all analysis results
        visualize_community_topics(community_subgraph, community_id, df.set_index('paperId'))

    
    print("\n" + "="*80)

# --- Main Execution ---
if __name__ == "__main__":
    combined_analysis(INPUT_GRAPH_FILE, INPUT_EMBEDDING_CSV)
