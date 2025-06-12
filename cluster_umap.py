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
INPUT_GRAPHML_FILE = "final_v2_graph.graphml" # <-- IMPORTANT: SET YOUR GRAPHML FILENAME HERE
TOP_N_CLUSTERS = 20
ITEMS_PER_CLUSTER = 10

# --- UMAP Configuration ---
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = 'cosine'

# --- HDBSCAN Clustering Configuration ---
MIN_CLUSTER_SIZE = 60
MIN_SAMPLES = 1
# up -> conservative | down -> liberal

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

# --- NEW FUNCTION TO CALCULATE PAGERANK FROM GRAPHML ---
def calculate_pagerank_from_graphml(df, graphml_path):
    """
    Loads a pre-built graph from a GraphML file, calculates PageRank,
    and adds it as a new column to the dataframe.
    """
    logging.info(f"Loading graph from {graphml_path}...")
    try:
        G = nx.read_graphml(graphml_path)
    except FileNotFoundError:
        logging.error(f"GraphML file not found: '{graphml_path}'. Please check the filename and path.")
        # Add an empty pagerank column so the script doesn't fail later
        df['pagerank'] = 0
        return df
    except Exception as e:
        logging.error(f"An error occurred while reading the GraphML file: {e}")
        df['pagerank'] = 0
        return df

    # Check that the dataframe has the required 'paper_id' column
    if 'paper_id' not in df.columns:
        logging.error("The CSV dataframe must have a 'paper_id' column to map PageRank scores.")
        df['pagerank'] = 0
        return df

    logging.info(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    logging.info("Calculating PageRank...")
    
    # Calculate PageRank
    pagerank_scores = nx.pagerank(G, alpha=0.85)

    # Map the scores back to the dataframe using the 'paper_id' column
    # Assumes node IDs in the graph match the 'paper_id' values in the CSV
    df['pagerank'] = df['paper_id'].map(pagerank_scores)
    
    # Fill any papers that were not in the graph with a low score
    df['pagerank'].fillna(0, inplace=True)
    
    logging.info("PageRank calculation complete.")
    return df

def cluster_with_umap_hdbscan(csv_path, graphml_path):
    """
    Loads embeddings, calculates PageRank from a GraphML file, reduces 
    dimensionality, performs clustering, and analyzes the results.
    """
    logging.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}. Please provide a valid input file.")
        return

    # --- COMPUTE PAGERANK FROM THE GRAPHML FILE ---
    df = calculate_pagerank_from_graphml(df, graphml_path)

    # --- Prepare the Original Embedding Data ---
    logging.info("Preparing original high-dimensional embedding vectors...")
    df.dropna(subset=['embedding'], inplace=True)
    df = df[df['embedding'].str.strip().isin(['', 'null', '[]']) == False]

    if df.empty:
        logging.warning("No papers with valid embeddings found. Cannot perform clustering.")
        return

    logging.info(f"Parsing {len(df)} embedding strings into vectors...")
    original_embeddings = np.array(df['embedding'].progress_apply(json.loads).tolist())
    logging.info("Original embedding data prepared successfully.")

    # --- Apply UMAP for Dimensionality Reduction ---
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

    # --- Perform HDBSCAN Clustering ---
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

    # --- Analyze and Display Results ---
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
            
        print("  Sample Papers (by PageRank):")
        for j, row in enumerate(cluster_df.sort_values(by='pagerank', ascending=False).head(ITEMS_PER_CLUSTER).itertuples()):
            title = str(row.title)
            display_title = (title[:80] + '...') if len(title) > 80 else title
            print(f"    - ({row.year}) {display_title}")
            
        print("")

if __name__ == "__main__":
    cluster_with_umap_hdbscan(INPUT_CSV_FILE, INPUT_GRAPHML_FILE)