import pandas as pd
import networkx as nx
import logging

# --- Configuration ---
# 1. Path to your graph file to get PageRank scores.
INPUT_GRAPH_FILE = "final_v2_graph.graphml"

# 2. Path to your main dataset to get citation counts.
INPUT_CSV_FILE = "final_v2_papers.csv"

# 3. How many papers to select from each list (PageRank and Citations).
TOP_N = 2000

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_combined_top_papers(graph_path, csv_path, top_n):
    """
    Finds the top N papers by PageRank and by citation count, then combines
    them into a single unique list of paper IDs.
    """
    # --- Step 1: Get Top Papers by PageRank ---
    logging.info(f"Loading graph from {graph_path} to calculate PageRank...")
    try:
        G = nx.read_graphml(graph_path)
        pagerank_scores = nx.pagerank(G, weight='weight')
        # Sort by score and get the paper IDs
        top_pagerank_ids = [
            pid for pid, score in sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
        ][:top_n]
        logging.info(f"Identified Top {len(top_pagerank_ids)} papers by PageRank.")
    except Exception as e:
        logging.error(f"Could not process graph file: {e}")
        top_pagerank_ids = []

    # --- Step 2: Get Top Papers by Citation Count ---
    logging.info(f"Loading data from {csv_path} to find most cited papers...")
    try:
        df = pd.read_csv(csv_path)
        # Ensure citationCount is a numeric type for sorting
        df['citationCount'] = pd.to_numeric(df['citationCount'], errors='coerce').fillna(0)
        top_citation_df = df.sort_values(by='citationCount', ascending=False).head(top_n)
        top_citation_ids = top_citation_df['paperId'].tolist()
        logging.info(f"Identified Top {len(top_citation_ids)} papers by Citation Count.")
    except Exception as e:
        logging.error(f"Could not process CSV file: {e}")
        top_citation_ids = []
        
    # --- Step 3: Combine and Deduplicate the Lists ---
    # Using a set automatically handles duplicates
    print("\n" + "="*80)  
    print("LANDMARK_PAPER_IDS_1 = [")
    for i, paper_id in enumerate(top_pagerank_ids):
        # Add a comma unless it's the last item
        comma = "," if i < len(top_pagerank_ids) - 1 else ""
        print(f'    "{paper_id}"{comma}')
    print("]")
 
    # print("LANDMARK_PAPER_IDS_2 = [")
    # for i, paper_id in enumerate(top_citation_ids):
    #     # Add a comma unless it's the last item
    #     comma = "," if i < len(top_citation_ids) - 1 else ""
    #     print(f'    "{paper_id}"{comma}')
    # print("]")
    # print("\n" + "="*80)  


if __name__ == "__main__":
    get_combined_top_papers(INPUT_GRAPH_FILE, INPUT_CSV_FILE, TOP_N)