import pandas as pd
import numpy as np
import json
import logging
from tqdm import tqdm
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import networkx as nx
from hdbscan.validity import validity_index
from itertools import product

# --- Configuration ---
INPUT_CSV_FILE = "final_v2_papers.csv"
INPUT_GRAPHML_FILE = "final_v2_graph.graphml"
OUTPUT_REPORT_FILE = "analysis_report_adaptive_extended.md"
OUTPUT_DATA_FILE = "analysis_data_adaptive_extended.json"
   
# Time window settings
START_YEAR = 2010
END_YEAR = 2026
WINDOW_SIZE = 3
SLIDE_INCREMENT = 1

# --- EXTENDED Parameter Search Space Configuration ---
PARAMETER_SEARCH_SPACE = {
    'n_neighbors': [10, 15, 25, 35, 50],
    'min_cluster_size': [15, 25, 35, 50, 75],  
    'min_samples': [5, 10, 15, 25],
    'cluster_selection_epsilon': [0.0, 0.25, 0.5]
}

# --- Analysis settings ---
TOP_N_PAPERS = 20
TOP_N_KEYWORDS = 15

# --- Logging and Progress Bar Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()


def find_best_clustering_params(embeddings, search_space):
    """
    Tests all combinations of UMAP/HDBSCAN parameters and returns the one
    that produces the highest DBCV score.
    """
    best_score = -1.0
    best_params = {}

    param_combinations = list(product(
        search_space['n_neighbors'],
        search_space['min_cluster_size'],
        search_space['min_samples'],
        search_space['cluster_selection_epsilon']
    ))
    
    # Use a progress bar for the search itself, as it's the slowest part
    for n_neighbors, min_cluster_size, min_samples, epsilon in tqdm(param_combinations, desc="Searching Params"):
        if min_samples > min_cluster_size:
            continue
        try:
            umap_params = {'n_neighbors': n_neighbors, 'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'}
            hdbscan_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples, 
                              'metric': 'euclidean', 'cluster_selection_epsilon': epsilon}
            
            reducer = umap.UMAP(**umap_params, random_state=42).fit(embeddings)
            reduced_embeddings = reducer.transform(embeddings).astype(np.float64)
            clusterer = hdbscan.HDBSCAN(**hdbscan_params).fit(reduced_embeddings)
            labels = clusterer.labels_

            if len(np.unique(labels)) > 2:
                score = validity_index(reduced_embeddings, labels, metric='euclidean')
                if score > best_score:
                    best_score = score
                    best_params = {
                        'n_neighbors': n_neighbors,
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'cluster_selection_epsilon': epsilon,
                        'dbcv_score': score
                    }
        except Exception:
            continue # Some combinations may fail, this is expected
            
    return best_params

def get_topic_data(df_window, num_keywords, avg_probabilities):
    if 'cluster_id' not in df_window.columns or df_window['cluster_id'].max() < 0: return []
    topics = []
    cluster_ids = [cid for cid in df_window['cluster_id'].unique() if cid != -1]
    for cid in cluster_ids:
        cluster_df = df_window[df_window['cluster_id'] == cid]
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(cluster_df['title'].dropna())
            feature_names = vectorizer.get_feature_names_out()
            top_keywords_list = [feature_names[j] for j in tfidf_matrix.sum(axis=0).argsort()[0, ::-1].tolist()[0][:num_keywords]]
            topics.append({'cluster_id': int(cid), 'size': len(cluster_df), 'keywords': top_keywords_list, 'confidence': avg_probabilities.get(cid, 0)})
        except ValueError:
            continue
    return sorted(topics, key=lambda x: x['size'], reverse=True)

def write_window_summary_to_file(file_handle, window_data):
    window_start, window_end = window_data['window_start'], window_data['window_end']
    file_handle.write(f"\n# Analysis for Window: {window_start}-{window_end}\n")
    file_handle.write(f"**Stats:** Papers: {window_data['paper_count']}, Edges: {window_data['edge_count']}\n")
    file_handle.write(f"**Best Found DBCV Score:** {window_data['dbcv_score']:.4f}\n")
    optimal_p = window_data['optimal_params']
    file_handle.write(f"**Optimal Parameters:** `n_neighbors`: {optimal_p['n_neighbors']}, `min_cluster_size`: {optimal_p['min_cluster_size']}, `min_samples`: {optimal_p['min_samples']}, `epsilon`: {optimal_p['cluster_selection_epsilon']}\n\n")
    
    file_handle.write("### Dominant Topics\n\n")
    for topic in window_data['topics']:
        file_handle.write(f"- **Topic (ID: {topic['cluster_id']}, Size: {topic['size']}, Confidence: {topic['confidence']:.4f}):** {', '.join(topic['keywords'])}\n")
    file_handle.write("\n")

    for metric_key, metric_name in [('by_citation', 'Citation Count'), ('by_pagerank', 'PageRank'), ('by_authority', 'HITS Authority')]:
        file_handle.write(f"### Top {TOP_N_PAPERS} Papers by {metric_name}\n\n")
        for paper in window_data['top_papers'][metric_key]:
            display_title = (paper['title'][:75] + '...') if len(paper['title']) > 75 else paper['title']
            file_handle.write(f"- ({metric_name}: {paper['score']:.4f}) {display_title}\n")
        file_handle.write("\n")

def process_time_window(G, df_master, window_start, window_end):
    logging.info(f"\n--- Processing window: {window_start}-{window_end} ---")
    df_window = df_master[(df_master['year'] >= window_start) & (df_master['year'] <= window_end)].copy()
    df_with_embeddings = df_window.dropna(subset=['embedding'])
    df_with_embeddings = df_with_embeddings[df_with_embeddings['embedding'].str.strip().isin(['', 'null', '[]']) == False]

    if len(df_with_embeddings) < 20:
        logging.warning(f"Skipping window {window_start}-{window_end}: Not enough papers with embeddings.")
        return None

    embeddings = np.array(df_with_embeddings['embedding'].progress_apply(json.loads).tolist())
    optimal_params = find_best_clustering_params(embeddings, PARAMETER_SEARCH_SPACE)
    
    if not optimal_params:
        logging.warning(f"Could not find optimal parameters for window {window_start}-{window_end}.")
        return None
    logging.info(f"Optimal params found for {window_start}-{window_end}: {optimal_params}")

    umap_params = {'n_neighbors': optimal_params['n_neighbors'], 'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'}
    hdbscan_params = {'min_cluster_size': optimal_params['min_cluster_size'], 'min_samples': optimal_params['min_samples'], 'metric': 'euclidean', 'cluster_selection_epsilon': optimal_params['cluster_selection_epsilon']}

    reducer = umap.UMAP(**umap_params, random_state=42).fit(embeddings)
    reduced_embeddings = reducer.transform(embeddings).astype(np.float64)
    clusterer = hdbscan.HDBSCAN(**hdbscan_params).fit(reduced_embeddings)
    df_window.loc[df_with_embeddings.index, 'cluster_id'] = clusterer.labels_
    df_window.loc[df_with_embeddings.index, 'probability'] = clusterer.probabilities_
    df_window['cluster_id'] = df_window.get('cluster_id', -1).fillna(-1).astype(int)

    nodes_in_window = df_window['paperId'].tolist()
    G_window = nx.DiGraph(G.subgraph(nodes_in_window))
    pagerank_scores = nx.pagerank(G_window, alpha=0.85)
    _, authority_scores = nx.hits(G_window, max_iter=500)
    df_window['pagerank'] = df_window['paperId'].map(pagerank_scores).fillna(0)
    df_window['authority_score'] = df_window['paperId'].map(authority_scores).fillna(0)
    
    valid_clusters = df_window[df_window['cluster_id'] != -1]
    avg_probabilities = valid_clusters.groupby('cluster_id')['probability'].mean().to_dict()
    topic_data = get_topic_data(df_window, TOP_N_KEYWORDS, avg_probabilities)

    top_papers_data = {}
    for metric, name in [('citation_count', 'by_citation'), ('pagerank', 'by_pagerank'), ('authority_score', 'by_authority')]:
        top_list = []
        sorted_df = df_window.sort_values(by=metric, ascending=False).head(TOP_N_PAPERS)
        for _, row in sorted_df.iterrows():
            top_list.append({'title': row.get('title', 'N/A'), 'score': row[metric]})
        top_papers_data[name] = top_list
        
    return {
        "window_start": window_start, "window_end": window_end,
        "paper_count": G_window.number_of_nodes(), "edge_count": G_window.number_of_edges(),
        "dbcv_score": optimal_params['dbcv_score'],
        "optimal_params": {k:v for k,v in optimal_params.items() if k != 'dbcv_score'},
        "topics": topic_data, "top_papers": top_papers_data
    }

def main():
    logging.info("Loading master data files...")
    try:
        df_master = pd.read_csv(INPUT_CSV_FILE)
        if 'citationCount' in df_master.columns:
            df_master.rename(columns={'citationCount': 'citation_count'}, inplace=True)
        G_master = nx.read_graphml(INPUT_GRAPHML_FILE)
    except FileNotFoundError as e:
        logging.error(f"Fatal error loading files: {e}. Exiting.")
        return

    all_windows_data = []
    with open(OUTPUT_REPORT_FILE, 'w', encoding='utf-8') as report_file:
        report_file.write("# Dynamic Analysis of Scientific Literature (Adaptive Parameters, Extended Search)\n")
        
        for year in range(START_YEAR, END_YEAR - WINDOW_SIZE + 2, SLIDE_INCREMENT):
            window_start, window_end = year, year + WINDOW_SIZE - 1
            if window_end > END_YEAR: break
            window_data = process_time_window(G_master, df_master, window_start, window_end)
            if window_data:
                write_window_summary_to_file(report_file, window_data)
                all_windows_data.append(window_data)
    logging.info(f"Human-readable report saved to {OUTPUT_REPORT_FILE}")

    with open(OUTPUT_DATA_FILE, 'w', encoding='utf-8') as json_file:
        json.dump(all_windows_data, json_file, indent=2)
    logging.info(f"Machine-readable data saved to {OUTPUT_DATA_FILE}")

if __name__ == "__main__":
    main()