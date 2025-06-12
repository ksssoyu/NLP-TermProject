import pandas as pd
import numpy as np
import json
import logging
from tqdm import tqdm
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import networkx as nx

# --- Configuration ---
INPUT_CSV_FILE = "final_v2_papers.csv"
INPUT_GRAPHML_FILE = "final_v2_graph.graphml"
TOP_N_CLUSTERS = 10
ITEMS_PER_CLUSTER = 8

# --- UMAP Configuration ---
UMAP_N_NEIGHBORS = 50
UMAP_N_COMPONENTS = 10
UMAP_MIN_DIST = 0.0
UMAP_METRIC = 'cosine'

# --- HDBSCAN Clustering Configuration ---
MIN_CLUSTER_SIZE = 400
MIN_SAMPLES = 400

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

# --- FUNCTION TO ADD GRAPH METRICS ---
def add_graph_metrics_from_graphml(df, graphml_path):
    """
    Loads a pre-built graph, calculates PageRank, and extracts the citation count
    attribute from each node.
    """
    logging.info(f"Loading graph from {graphml_path}...")
    try:
        G = nx.read_graphml(graphml_path)
    except FileNotFoundError:
        logging.error(f"GraphML file not found: '{graphml_path}'.")
        df['pagerank'] = 0
        df['citation_count'] = 0
        return df
    except Exception as e:
        logging.error(f"An error occurred while reading the GraphML file: {e}")
        df['pagerank'] = 0
        df['citation_count'] = 0
        return df

    # --- FIX 1: Use 'paperId' (capital I) to match the graph construction ---
    if 'paperId' not in df.columns:
        logging.error("The CSV dataframe must have a 'paperId' column to map graph metrics.")
        df['pagerank'] = 0
        df['citation_count'] = 0
        return df

    logging.info(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    logging.info("Calculating PageRank...")
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    # --- FIX 1: Map using 'paperId' ---
    df['pagerank'] = df['paperId'].map(pagerank_scores).fillna(0)
    
    # --- FIX 2: Use 'citationCount' (capital C) to match the graph attribute ---
    logging.info("Extracting citation count from graph node attributes...")
    citation_counts_from_graph = nx.get_node_attributes(G, 'citationCount')
    
    if not citation_counts_from_graph:
        logging.warning("No 'citationCount' attribute found in graph nodes. Defaulting to 0.")
        df['citation_count'] = 0
    else:
        # --- FIX 1 & 2: Map using 'paperId' and the corrected attribute data ---
        df['citation_count'] = df['paperId'].map(citation_counts_from_graph).fillna(0).astype(int)
    
    logging.info("Graph metrics processing complete.")
    return df


def cluster_with_umap_hdbscan(csv_path, graphml_path):
    """
    Loads data, adds graph metrics, performs clustering, and analyzes results.
    """
    logging.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}. Please provide a valid input file.")
        return

    df = add_graph_metrics_from_graphml(df, graphml_path)

    logging.info("Preparing original high-dimensional embedding vectors...")
    df.dropna(subset=['embedding'], inplace=True)
    df = df[df['embedding'].str.strip().isin(['', 'null', '[]']) == False]

    if df.empty:
        logging.warning("No papers with valid embeddings found. Cannot perform clustering.")
        return

    logging.info(f"Parsing {len(df)} embedding strings into vectors...")
    original_embeddings = np.array(df['embedding'].progress_apply(json.loads).tolist())
    logging.info("Original embedding data prepared successfully.")

    logging.info(f"Applying UMAP to reduce dimensions to {UMAP_N_COMPONENTS}...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(original_embeddings)
    logging.info("Dimensionality reduction complete.")

    logging.info(f"Running HDBSCAN on the new {UMAP_N_COMPONENTS}-dimensional data...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        gen_min_span_tree=True
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    df['cluster_id'] = cluster_labels
    logging.info("Clustering finished.")

    # --- 1. Analyze and Display Cluster Results ---
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise_points = np.sum(cluster_labels == -1)
    
    print("\n" + "="*80)
    print("  UMAP + HDBSCAN Clustering Summary")
    print("="*80)
    print(f"Number of clusters found: {num_clusters}")
    print(f"Number of noise points (outliers): {num_noise_points}")
    print("="*80)

    clusters = df[df['cluster_id'] != -1].groupby('cluster_id')
    sorted_clusters = sorted(clusters, key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nDisplaying Top {min(TOP_N_CLUSTERS, num_clusters)} Largest Clusters:\n")

    for i, (cluster_id, cluster_df) in enumerate(sorted_clusters[:TOP_N_CLUSTERS]):
        print(f"--- Cluster {i+1} (ID: {cluster_id}): {len(cluster_df)} papers ---")
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_df['title'].dropna())
            feature_names = vectorizer.get_feature_names_out()
            top_keywords = [feature_names[j] for j in tfidf_matrix.sum(axis=0).argsort()[0, ::-1].tolist()[0][:ITEMS_PER_CLUSTER]]
            print(f"  Topic Keywords: {', '.join(top_keywords)}")
        except ValueError:
            print("  Topic Keywords: Not enough titles to determine keywords.")
            
        print("  Sample Papers (Sorted by Citation Count):")
        for j, row in enumerate(cluster_df.sort_values(by='citation_count', ascending=False).head(ITEMS_PER_CLUSTER).itertuples()):
            title = str(row.title)
            display_title = (title[:70] + '...') if len(title) > 70 else title
            print(f"    - (Cites: {row.citation_count}, PR: {row.pagerank:.4f}, Year: {row.year}) {display_title}")
            
        print("")

    # --- 2. Analyze and Display Outliers ---
    print("\n" + "="*80)
    print("  Top Outlier Papers (Not in Any Cluster)")
    print("="*80)

    outliers_df = df[df['cluster_id'] == -1]

    if not outliers_df.empty:
        top_outliers = outliers_df.sort_values(by='citation_count', ascending=False).head(100)
        
        print(f"Displaying top {len(top_outliers)} of {len(outliers_df)} outlier papers (sorted by citation count):\n")
        
        for index, row in top_outliers.iterrows():
            title = str(row.title)
            display_title = (title[:70] + '...') if len(title) > 70 else title
            print(f"  - (Cites: {row.citation_count}, PR: {row.pagerank:.4f}, Year: {row.year}) {display_title}")
    else:
        print("No outlier papers found.")
    
    print("")

if __name__ == "__main__":
    cluster_with_umap_hdbscan(INPUT_CSV_FILE, INPUT_GRAPHML_FILE)