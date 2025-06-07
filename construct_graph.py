import pandas as pd
import networkx as nx
import json
import logging
from tqdm import tqdm

# --- Configuration ---
# The path to the dataset collected by the crawler.
INPUT_CSV_FILE = "nlp_papers_dataset_v6_with_isInfluential.csv"
# The path where you want to save the final graph file.
OUTPUT_GRAPH_FILE = "citation_graph.graphml"

# --- Edge Weight Configuration ---
WEIGHT_CONFIG = {
    'base_weight': 1.0,
    'influential_multiplier': 2.0,
    'intent_scores': {
        'methodology': 1.5,
        'extension': 1.2,
        'comparison': 0.8,
        'background': 0.5,
        'result': 0.7,
    },
    'unknown_intent_score': 0.2
}

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Functions ---

def calculate_edge_weight(intents, is_influential, config):
    """
    Calculates the weight for a single citation edge based on the configuration.
    """
    total_weight = config.get('base_weight', 1.0)
    intent_score_sum = 0
    if intents:
        for intent in intents:
            intent_score_sum += config['intent_scores'].get(intent, config.get('unknown_intent_score', 0.1))
    total_weight += intent_score_sum
    if is_influential:
        total_weight *= config.get('influential_multiplier', 1.0)
    return total_weight

def build_citation_graph(csv_path, weight_config):
    """
    Loads paper data from a CSV and builds a weighted directed graph,
    ensuring edges only connect nodes that exist within the dataset.
    """
    logging.info(f"Loading data from {csv_path}...")
    try:
        # Keep NaN values as placeholders rather than converting to empty strings
        df = pd.read_csv(csv_path).where(pd.notna, None)
        if df.empty:
            logging.warning(f"Input CSV file '{csv_path}' is empty. No graph will be built.")
            return nx.DiGraph() # Return an empty graph to be handled later
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}. Please ensure the crawler script ran successfully and created this file.")
        return None
    except pd.errors.EmptyDataError:
        logging.warning(f"Input CSV file '{csv_path}' is empty. No graph will be built.")
        return nx.DiGraph()

    logging.info(f"Loaded {len(df)} papers. Building graph...")
    
    G = nx.DiGraph()
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Adding nodes"):
        node_id = row['paperId']
        if pd.isna(node_id): continue
        
        # --- IMPORTANT FIX: Sanitize attributes to remove None values ---
        # Convert potential None/NaN values to types GraphML supports (str, int, float, bool).
        year_val = row.get('year')
        cit_count_val = row.get('citationCount')

        attributes = {
            'title': str(row.get('title', '')),
            'year': int(year_val) if pd.notna(year_val) else 0,
            'authors': str(row.get('authors', '')),
            'citationCount': int(cit_count_val) if pd.notna(cit_count_val) else 0,
            's2FieldsOfStudy': str(row.get('s2FieldsOfStudy', '')),
            'publicationVenueName': str(row.get('publicationVenueName', ''))
        }
        G.add_node(node_id, **attributes)

    skipped_edges_count = 0
    total_potential_edges = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Adding edges"):
        source_node_id = row['paperId']
        if pd.isna(source_node_id): continue

        try:
            if row['references_data'] is not None:
                references = json.loads(row['references_data'])
            else:
                references = []
        except (json.JSONDecodeError, TypeError):
            logging.warning(f"Could not parse references_data for paper {source_node_id}. Skipping.")
            references = []

        for ref in references:
            target_node_id = ref.get('paperId')
            total_potential_edges += 1
            if not target_node_id:
                skipped_edges_count += 1
                continue
            
            if G.has_node(target_node_id):
                intents = ref.get('intents', [])
                is_influential = ref.get('isInfluential', False)
                weight = calculate_edge_weight(intents, is_influential, weight_config)
                
                G.add_edge(
                    source_node_id,
                    target_node_id,
                    weight=weight,
                    intents=','.join(intents),
                    isInfluential=is_influential
                )
            else:
                skipped_edges_count += 1

    logging.info("Graph construction complete.")
    logging.info(f"Skipped {skipped_edges_count} edges out of {total_potential_edges} potential edges because the target paper was not in the collected dataset.")
    return G

# --- Main Execution ---
if __name__ == "__main__":
    citation_graph = build_citation_graph(INPUT_CSV_FILE, WEIGHT_CONFIG)

    if citation_graph is not None:
        num_nodes = citation_graph.number_of_nodes()
        
        if num_nodes > 0:
            num_edges = citation_graph.number_of_edges()
            
            print("\n" + "="*40)
            print("Graph Construction Summary")
            print("="*40)
            print(f"Number of nodes (papers): {num_nodes}")
            print(f"Number of edges (citations): {num_edges}")
            isolated_nodes = len(list(nx.isolates(citation_graph)))
            print(f"Number of isolated nodes: {isolated_nodes}")

            try:
                logging.info(f"Saving graph with {num_nodes} nodes to {OUTPUT_GRAPH_FILE}...")
                nx.write_graphml(citation_graph, OUTPUT_GRAPH_FILE)
                logging.info("Graph saved successfully.")
            except Exception as e:
                logging.error(f"Failed to save graph to file: {e}")
            
            print("="*40)
        else:
            logging.warning("Graph was built but contains no nodes. This is likely because the input CSV was empty.")
            logging.warning(f"An empty graph file was NOT created at '{OUTPUT_GRAPH_FILE}'.")
    else:
        logging.error("Graph building failed due to an error. Please check logs.")
