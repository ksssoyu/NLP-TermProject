import pandas as pd
import networkx as nx
import json
import logging
import time
import requests
from tqdm import tqdm

'''
EXPLICIT USE ONLY TO FETCH DATA FROM SPECIFIC ENDPOINT AT SPECIFIED OFFSET AND LIMIT
'''

# --- Configuration ---
# The ID of the paper you want to add references to.
PAPER_ID_TO_UPDATE = "1b6e810ce0afd0dd093f789d2b2742d047e316d5" 

# --- NEW: Specify the offset and limit for fetching references ---
REFERENCES_OFFSET = 0  # The starting point of the reference list to fetch (e.g., start after the first 100)
REFERENCES_LIMIT = 100    # The maximum number of references to fetch in this run

# Your Semantic Scholar API Key
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx"
API_DELAY_SECONDS = 1.0

# --- File Paths (ensure these are correct) ---
INPUT_OUTPUT_CSV_FILE = "nlp_papers_dataset_v6_with_isInfluential.csv"
OUTPUT_GRAPH_FILE = "citation_graph.graphml"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Edge Weight Configuration (from construct_graph.py) ---
WEIGHT_CONFIG = {
    'base_weight': 1.0,
    'influential_multiplier': 2.0,
    'intent_scores': { 'methodology': 1.5, 'extension': 1.2, 'comparison': 0.8, 'background': 0.5, 'result': 0.7 },
    'unknown_intent_score': 0.2
}


# --- Helper Functions ---

def make_single_api_request(url, params, request_type="generic"):
    """Makes a single, robust request to the Semantic Scholar API."""
    headers = {}
    if API_KEY and API_KEY != "YOUR_SEMANTIC_SCHOLAR_API_KEY": headers['X-API-KEY'] = API_KEY
    try:
        time.sleep(API_DELAY_SECONDS)
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"API request failed for {url} with params {params}: {e}")
    return None

def fetch_connections_with_offset_limit(paper_id, conn_type, offset, limit):
    """
    MODIFIED: Fetches a slice of connections (references or citations) for a paper.
    """
    fields = f"intents,isInfluential,{'citingPaper' if conn_type == 'citations' else 'citedPaper'}.paperId"
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{conn_type}"
    params = {'fields': fields, 'offset': offset, 'limit': limit}
    
    logging.info(f"Fetching {conn_type} for {paper_id} with offset={offset} and limit={limit}")
    resp = make_single_api_request(base_url, params, f"{conn_type} fetch")
    
    if not resp or not resp.get('data'):
        logging.warning("No data returned from API for this slice.")
        return []
        
    return resp.get('data', [])

def process_connections(raw_conns, conn_type):
    """Processes raw connection data into a list of dictionaries."""
    processed = []
    key = "citingPaper" if conn_type == "citations" else "citedPaper"
    for item in raw_conns:
        paper_info = item.get(key, {})
        paper_id = paper_info.get('paperId')
        if not paper_id: continue
        processed.append({
            'paperId': paper_id,
            'intents': sorted(list(set(item.get('intents', [])))),
            'isInfluential': item.get('isInfluential', False)
        })
    return processed

def calculate_edge_weight(intents, is_influential, config):
    """Calculates the weight for a citation edge."""
    total_weight = config.get('base_weight', 1.0)
    intent_score_sum = sum(config['intent_scores'].get(intent, config.get('unknown_intent_score', 0.1)) for intent in intents)
    total_weight += intent_score_sum
    if is_influential:
        total_weight *= config.get('influential_multiplier', 1.0)
    return total_weight

def build_graph_from_df(df, weight_config):
    """Rebuilds the entire graph from the provided DataFrame."""
    logging.info("Re-building graph from updated DataFrame...")
    G = nx.DiGraph()
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Adding nodes"):
        node_id = row['paperId']
        if pd.isna(node_id): continue
        attributes = {
            'title': str(row.get('title', '')), 'year': int(row.get('year', 0)),
            'citationCount': int(row.get('citationCount', 0))
        }
        G.add_node(node_id, **attributes)

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Adding edges"):
        source_node_id = row['paperId']
        if pd.isna(source_node_id): continue
        try:
            references = json.loads(row['references_data']) if pd.notna(row['references_data']) else []
        except (json.JSONDecodeError, TypeError):
            references = []
        for ref in references:
            target_node_id = ref.get('paperId')
            if target_node_id and G.has_node(target_node_id):
                weight = calculate_edge_weight(ref.get('intents', []), ref.get('isInfluential', False), weight_config)
                G.add_edge(source_node_id, target_node_id, weight=weight)
    return G

# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Attempting to add references to paper '{PAPER_ID_TO_UPDATE}' in '{INPUT_OUTPUT_CSV_FILE}'.")
    
    # 1. Load existing data
    try:
        df = pd.read_csv(INPUT_OUTPUT_CSV_FILE)
    except FileNotFoundError:
        logging.critical(f"FATAL: Input file not found: {INPUT_OUTPUT_CSV_FILE}. Cannot proceed.")
        exit()

    # 2. Find the paper to update
    paper_index = df.index[df['paperId'] == PAPER_ID_TO_UPDATE].tolist()
    if not paper_index:
        logging.critical(f"FATAL: Paper {PAPER_ID_TO_UPDATE} not found in the dataset. Please add it first.")
        exit()
    
    paper_index = paper_index[0]
    logging.info(f"Found paper '{df.loc[paper_index, 'title']}' at index {paper_index}.")

    # 3. Fetch the new slice of reference data
    new_references_raw = fetch_connections_with_offset_limit(PAPER_ID_TO_UPDATE, 'references', REFERENCES_OFFSET, REFERENCES_LIMIT)
    
    if new_references_raw:
        new_references_processed = process_connections(new_references_raw, 'references')
        
        # 4. Load existing reference data and merge with new data
        try:
            existing_references = json.loads(df.loc[paper_index, 'references_data']) if pd.notna(df.loc[paper_index, 'references_data']) else []
        except (json.JSONDecodeError, TypeError):
            logging.warning("Could not parse existing reference data. Starting with an empty list.")
            existing_references = []

        # Create a set of existing reference IDs to prevent duplicates
        existing_ref_ids = {ref['paperId'] for ref in existing_references}
        
        # Add only the truly new references
        unique_new_refs = [ref for ref in new_references_processed if ref['paperId'] not in existing_ref_ids]
        
        if not unique_new_refs:
            logging.warning("The fetched references are already present in the dataset. No changes made.")
        else:
            logging.info(f"Adding {len(unique_new_refs)} new, unique references to the paper.")
            combined_references = existing_references + unique_new_refs
            
            # 5. Update the DataFrame and save it
            df.loc[paper_index, 'references_data'] = json.dumps(combined_references)
            try:
                df.to_csv(INPUT_OUTPUT_CSV_FILE, index=False)
                logging.info(f"Successfully updated CSV file '{INPUT_OUTPUT_CSV_FILE}'.")
            except Exception as e:
                logging.error(f"Failed to save updated CSV file: {e}")

            # 6. Re-build and save the graph
            final_graph = build_graph_from_df(df, WEIGHT_CONFIG)
            logging.info(f"Graph rebuilt with {final_graph.number_of_nodes()} nodes and {final_graph.number_of_edges()} edges.")
            try:
                nx.write_graphml(final_graph, OUTPUT_GRAPH_FILE)
                logging.info(f"Successfully saved updated graph to '{OUTPUT_GRAPH_FILE}'.")
            except Exception as e:
                logging.error(f"Failed to save the new graph file: {e}")
    else:
        logging.info("No new reference data was fetched. No changes made to the dataset or graph.")