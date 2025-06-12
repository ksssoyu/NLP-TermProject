import pandas as pd
import networkx as nx
import logging

# --- Configuration ---
# IMPORTANT: Set the path to your GraphML file here
INPUT_GRAPHML_FILE = "final_v2_graph.graphml"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_papers_without_embedding(graphml_path):
    """
    Loads a graph, finds all papers without an embedding field, and prints
    them in order of their citation count.
    """
    logging.info(f"Loading graph from {graphml_path}...")
    try:
        G = nx.read_graphml(graphml_path)
    except FileNotFoundError:
        logging.error(f"GraphML file not found: '{graphml_path}'. Please check the filename and path.")
        return
    except Exception as e:
        logging.error(f"An error occurred while reading the GraphML file: {e}")
        return

    logging.info(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Convert node attributes to a pandas DataFrame ---
    # We extract the attribute dictionary for each node and include the node's ID
    all_node_data = []
    for node_id, data_dict in G.nodes(data=True):
        record = data_dict.copy()
        record['paper_id'] = node_id
        all_node_data.append(record)
    
    df = pd.DataFrame(all_node_data)

    if df.empty:
        logging.warning("No node data found in the graph.")
        return

    # --- Identify papers without a valid embedding ---
    if 'embedding' in df.columns:
        # Condition checks for null values, empty strings, or string representations of empty lists
        no_embedding_condition = df['embedding'].isnull() | (df['embedding'].astype(str).str.strip().isin(['', '[]', 'None']))
        papers_without_embedding = df[no_embedding_condition]
    else:
        # If the 'embedding' column doesn't exist at all, all papers are considered
        logging.warning("No 'embedding' attribute found in any node. Listing all papers.")
        papers_without_embedding = df.copy()

    if papers_without_embedding.empty:
        logging.info("All papers in the graph have a valid embedding field. No papers to list.")
        return
        
    logging.info(f"Found {len(papers_without_embedding)} papers without an embedding field.")
    
    # --- Calculate and map citation count (in-degree) ---
    logging.info("Calculating citation counts...")
    citation_counts = dict(G.in_degree())
    papers_without_embedding['citation_count'] = papers_without_embedding['paper_id'].map(citation_counts).fillna(0).astype(int)

    # --- Sort by citation count ---
    sorted_papers = papers_without_embedding.sort_values(by='citation_count', ascending=False)
    p = sorted_papers[:100]
    # --- Print the final list ---
    print("\n" + "="*80)
    print("  Papers Without an Embedding Field (Sorted by Citation Count)")
    print("="*80)

    for index, row in p.iterrows():
        # Use .get() to safely access attributes that might not exist for every node
        title = row.get('title', 'No Title Available')
        year = row.get('year', 'N/A')
        citation_count = row.get('citation_count', 0)
        
        display_title = (title[:85] + '...') if len(str(title)) > 85 else title
        print(f"  - (Citations: {citation_count}, Year: {year}) {display_title}")

if __name__ == "__main__":
    list_papers_without_embedding(INPUT_GRAPHML_FILE)