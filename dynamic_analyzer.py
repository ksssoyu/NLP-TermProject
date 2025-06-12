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

# Time window settings
START_YEAR = 2015
END_YEAR = 2027
WINDOW_SIZE = 3
SLIDE_INCREMENT = 1 # How many years to slide the window forward each step

# Analysis settings
TOP_N_PAPERS = 20
TOP_N_KEYWORDS = 15

# Algorithm parameters
UMAP_PARAMS = {'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'}
HDBSCAN_PARAMS = {'min_cluster_size': 20, 'min_samples': 20, 'metric': 'euclidean'}

# --- Logging and Progress Bar Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()


def get_top_keywords(df_window, num_keywords):
    """Extracts top keywords from the largest cluster in a given dataframe."""
    # This check is now robust because 'cluster_id' will not contain NaNs
    if 'cluster_id' not in df_window.columns or df_window['cluster_id'].max() < 0:
        return "No significant clusters found."

    top_cluster_id = df_window[df_window['cluster_id'] != -1]['cluster_id'].value_counts().idxmax()
    cluster_df = df_window[df_window['cluster_id'] == top_cluster_id]

    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(cluster_df['title'].dropna())
        feature_names = vectorizer.get_feature_names_out()
        top_keywords_list = [feature_names[j] for j in tfidf_matrix.sum(axis=0).argsort()[0, ::-1].tolist()[0][:num_keywords]]
        return ', '.join(top_keywords_list)
    except ValueError:
        return "Not enough titles in the cluster to determine keywords."

def print_window_summary(results_df, window_start, window_end, keywords, num_papers, num_edges):
    """Formats and prints the summary for a single time window."""
    print("\n" + "#"*80)
    print(f"# Analysis for Window: {window_start}-{window_end} (Papers: {num_papers}, Citations: {num_edges})")
    print("#"*80)
    print(f"\nDominant Topic Keywords: {keywords}\n")

    for metric, name in [('citation_count', 'Citation Count'), ('pagerank', 'PageRank'), ('authority_score', 'HITS Authority')]:
        print("-" * 80)
        print(f"Top {TOP_N_PAPERS} Papers by {name}")
        print("-" * 80)
        
        top_papers = results_df.sort_values(by=metric, ascending=False).head(TOP_N_PAPERS)
        for _, row in top_papers.iterrows():
            title = str(row.get('title', 'N/A'))
            display_title = (title[:75] + '...') if len(title) > 75 else title
            print(f"  - ({metric}: {row[metric]:.4f}) {display_title}")
        print("")


def process_time_window(G, df_master, window_start, window_end):
    """Analyzes a single time window of the graph."""
    logging.info(f"Processing window: {window_start}-{window_end}...")
    df_window = df_master[(df_master['year'] >= window_start) & (df_master['year'] <= window_end)].copy()
    
    if len(df_window) < HDBSCAN_PARAMS['min_cluster_size']:
        logging.warning(f"Skipping window {window_start}-{window_end}: Not enough papers ({len(df_window)}) for analysis.")
        return

    nodes_in_window = df_window['paperId'].tolist()
    G_window = nx.DiGraph(G.subgraph(nodes_in_window))

    pagerank_scores = nx.pagerank(G_window, alpha=0.85)
    hub_scores, authority_scores = nx.hits(G_window, max_iter=500)

    df_window['pagerank'] = df_window['paperId'].map(pagerank_scores).fillna(0)
    df_window['authority_score'] = df_window['paperId'].map(authority_scores).fillna(0)
    
    # Clustering
    df_with_embeddings = df_window.dropna(subset=['embedding'])
    df_with_embeddings = df_with_embeddings[df_with_embeddings['embedding'].str.strip().isin(['', 'null', '[]']) == False]
    
    keywords = "No valid embeddings found for this window."
    if not df_with_embeddings.empty:
        try:
            embeddings = np.array(df_with_embeddings['embedding'].progress_apply(json.loads).tolist())
            reducer = umap.UMAP(**UMAP_PARAMS, random_state=42).fit(embeddings)
            reduced_embeddings = reducer.transform(embeddings)
            clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS).fit(reduced_embeddings)
            df_window.loc[df_with_embeddings.index, 'cluster_id'] = clusterer.labels_
        except Exception as e:
            logging.error(f"Clustering failed for window {window_start}-{window_end}: {e}")
    
    # --- FIX: Fill any remaining NaN cluster_id's with -1 and ensure column is integer type ---
    # This prevents errors in any downstream function that uses this column.
    df_window['cluster_id'] = df_window.get('cluster_id', -1).fillna(-1).astype(int)

    keywords = get_top_keywords(df_window, TOP_N_KEYWORDS)

    print_window_summary(df_window, window_start, window_end, keywords, G_window.number_of_nodes(), G_window.number_of_edges())


def main():
    """Main function to orchestrate the dynamic analysis."""
    logging.info("Loading master data files...")
    try:
        df_master = pd.read_csv(INPUT_CSV_FILE)
        if 'citationCount' in df_master.columns:
            df_master.rename(columns={'citationCount': 'citation_count'}, inplace=True)
        G_master = nx.read_graphml(INPUT_GRAPHML_FILE)
    except FileNotFoundError as e:
        logging.error(f"Fatal error loading files: {e}. Exiting.")
        return

    for year in range(START_YEAR, END_YEAR - WINDOW_SIZE + 2, SLIDE_INCREMENT):
        window_start = year
        window_end = year + WINDOW_SIZE - 1
        if window_end > END_YEAR:
            break
        
        process_time_window(G_master, df_master, window_start, window_end)

if __name__ == "__main__":
    main()