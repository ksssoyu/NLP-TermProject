import pandas as pd
import numpy as np
import json
import logging
from tqdm import tqdm
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---
# The path to your final, enriched dataset that contains embeddings.
INPUT_CSV_FILE = "final_v2_papers.csv"
# The number of top clusters to display in the summary.
TOP_N_CLUSTERS = 15
# The number of top keywords/papers to show from each cluster.
ITEMS_PER_CLUSTER = 8

# --- HDBSCAN Clustering Configuration ---
# The minimum number of papers required to form a distinct cluster.
# This is the most important parameter to tune. Start with values between 5 and 15.
MIN_CLUSTER_SIZE = 500
# Controls how conservative the clustering is. Higher values mean more points are declared as noise.
# Can be left equal to MIN_CLUSTER_SIZE as a good starting point.
MIN_SAMPLES = 500

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

def cluster_by_embeddings(csv_path):
    """
    Loads paper embeddings from a CSV, performs HDBSCAN clustering, and analyzes the results.
    """
    logging.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}. Please provide a valid input file.")
        return

    # --- 1. Prepare the Embedding Data ---
    logging.info("Preparing embedding vectors...")
    # Drop rows where embedding is missing, as they cannot be clustered.
    df.dropna(subset=['embedding'], inplace=True)
    df = df[df['embedding'].str.strip() != 'null']
    df = df[df['embedding'].str.strip() != '[]']

    if df.empty:
        logging.warning("No papers with valid embeddings found. Cannot perform clustering.")
        return

    # Parse the JSON string into a list of vectors. This can be slow.
    logging.info(f"Parsing {len(df)} embedding strings into vectors...")
    embeddings = np.array(df['embedding'].progress_apply(json.loads).tolist())
    
    logging.info("Embedding data prepared successfully.")

    # --- 2. Perform HDBSCAN Clustering ---
    logging.info(f"Running HDBSCAN with min_cluster_size={MIN_CLUSTER_SIZE}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        gen_min_span_tree=True
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Add cluster labels back to the DataFrame for analysis
    df['cluster_id'] = cluster_labels
    logging.info("Clustering finished.")

    # --- 3. Analyze and Display Results ---
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise_points = np.sum(cluster_labels == -1)
    
    print("\n" + "="*80)
    print("  Embedding Clustering Summary (HDBSCAN)")
    print("="*80)
    print(f"Number of clusters found: {num_clusters}")
    print(f"Number of noise points (outliers): {num_noise_points}")
    print("="*80)

    # Group papers by cluster ID
    clusters = df[df['cluster_id'] != -1].groupby('cluster_id')
    
    # Sort clusters by size (number of papers)
    sorted_clusters = sorted(clusters, key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nDisplaying Top {min(TOP_N_CLUSTERS, num_clusters)} Largest Clusters:\n")

    for i, (cluster_id, cluster_df) in enumerate(sorted_clusters[:TOP_N_CLUSTERS]):
        print(f"--- Cluster {i+1} (ID: {cluster_id}): {len(cluster_df)} papers ---")
        
        # --- Generate Cluster Keywords using TF-IDF ---
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            # Fit TF-IDF on the titles within the current cluster
            tfidf_matrix = vectorizer.fit_transform(cluster_df['title'].dropna())
            feature_names = vectorizer.get_feature_names_out()
            # Get the top keywords for this cluster
            top_keywords = [feature_names[j] for j in tfidf_matrix.sum(axis=0).argsort()[0, ::-1].tolist()[0][:ITEMS_PER_CLUSTER]]
            print(f"  Topic Keywords: {', '.join(top_keywords)}")
        except ValueError:
            print("  Topic Keywords: Not enough titles to determine keywords.")
            
        print("  Sample Papers:")
        # Display the first few paper titles from the cluster
        for j, row in enumerate(cluster_df.head(ITEMS_PER_CLUSTER).itertuples()):
            title = str(row.title)
            display_title = (title[:80] + '...') if len(title) > 80 else title
            print(f"    - ({row.year}) {display_title}")
            
        print("") # Add a blank line for readability

# --- Main Execution ---
if __name__ == "__main__":
    cluster_by_embeddings(INPUT_CSV_FILE)