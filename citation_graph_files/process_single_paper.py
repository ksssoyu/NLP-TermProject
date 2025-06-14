import pandas as pd
import networkx as nx
import json
import logging
import time
import requests
from tqdm import tqdm

# --- Configuration ---
# The ID of the paper you want to fetch and add to your dataset.
PAPER_ID_TO_ADD = "f9a7175198a2c9f3ab0134a12a7e9e5369428e42" 

# Your Semantic Scholar API Key
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx"
API_DELAY_SECONDS = 1.0

# --- File Paths ---
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

# --- Relevance Filter Constants (from semantic_scholars_sync_fetch.py) ---
CORE_S2_NLP_FOS = ['natural language processing', 'computational linguistics']
TOP_NLP_VENUES = [
    'acl', 'emnlp', 'naacl', 'coling', 'eacl',
    'transactions of the association for computational linguistics', 'computational linguistics', 'lrec',
    'neurips', 'icml', 'iclr', 'aaai', 'ijcai'
]
BROADER_S2_NLP_FOS = ['artificial intelligence', 'linguistics', 'information retrieval', 'speech recognition', 'machine learning']
CS_FOS_CATEGORY = "computer science"


# --- Helper Functions ---

def make_single_api_request(url, params, request_type="generic"):
    """
    Makes a single, robust request to the Semantic Scholar API,
    with automatic retries for 429 rate limit errors.
    """
    headers = {}
    if API_KEY and API_KEY != "YOUR_SEMANTIC_SCHOLAR_API_KEY":
        headers['X-API-KEY'] = API_KEY
    
    max_retries = 3
    retry_delay = API_DELAY_SECONDS * 10  # Start with a longer delay
    retries = 0

    while retries < max_retries:
        try:
            # Always have a base delay for every attempt
            time.sleep(API_DELAY_SECONDS)
            
            logging.debug(f"Making API {request_type} request (Attempt {retries + 1}): URL={url}, Params={params}")
            response = requests.get(url, params=params, headers=headers)
            
            # This will raise an HTTPError for 4xx/5xx responses
            response.raise_for_status()
            
            # If successful, return the JSON and exit the loop
            return response.json()

        except requests.exceptions.HTTPError as e:
            # Check specifically for the 429 Rate Limit error
            if e.response.status_code == 429:
                retries += 1
                logging.warning(f"Rate limit (429) encountered for {url}. Waiting {retry_delay:.1f} seconds before retry {retries}/{max_retries}.")
                time.sleep(retry_delay)
                # Increase delay for subsequent retries
                retry_delay *= 2
                continue # Go to the next iteration of the while loop to retry
            else:
                # For any other HTTP error, log it and fail immediately
                logging.error(f"HTTP error {e.response.status_code} for {request_type} URL {url}: {e.response.text}")
                return None # Exit loop and function
                
        except Exception as e:
            # For non-HTTP errors (e.g., network issues), fail immediately
            logging.error(f"An unexpected error occurred for {url}: {e}")
            return None # Exit loop and function

    logging.error(f"Failed to fetch data for {url} after {max_retries} retries.")
    return None

def fetch_all_connections(paper_id, conn_type):
    all_items, offset = [], 0
    fields = f"intents,isInfluential,{'citingPaper' if conn_type == 'citations' else 'citedPaper'}.paperId"
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{conn_type}"
    while True:
        params = {'fields': fields, 'limit': 100, 'offset': offset}
        resp = make_single_api_request(base_url, params, f"{conn_type} fetch")
        if not resp or not resp.get('data'): break
        all_items.extend(resp.get('data', []))
        if resp.get('next') is None: break
        offset = resp['next']
    return all_items

def process_connections(raw_conns, conn_type):
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

def is_nlp_related_enhanced(api_response_data):
    """
    Checks if a paper is NLP-related using the same logic as the main crawler.
    """
    if not api_response_data: return False
    
    # Check for strong signals first
    if api_response_data.get('externalIds', {}).get('ACL'): return True
    
    venue_name = (api_response_data.get('publicationVenue') or {}).get('name', '').lower()
    if any(top_venue in venue_name for top_venue in TOP_NLP_VENUES): return True
    
    s2_categories = {field.get('category', '').lower() for field in api_response_data.get('s2FieldsOfStudy', [])}
    if CS_FOS_CATEGORY in s2_categories: return True
    if any(core_fos in s2_categories for core_fos in CORE_S2_NLP_FOS): return True
    
    # Check for weaker signals
    has_arxiv = bool(api_response_data.get('externalIds', {}).get('ArXiv'))
    broader_fos_count = sum(1 for fos in BROADER_S2_NLP_FOS if fos in s2_categories)
    
    if has_arxiv and broader_fos_count > 0: return True
    
    # If no criteria are met, it's not considered relevant.
    return False

def fetch_single_paper_data(paper_id):
    """Fetches all required data for a single paper, now with a relevance check."""
    logging.info(f"Fetching complete data for paper: {paper_id}")
    fields = "paperId,title,authors.name,year,citationCount,s2FieldsOfStudy,publicationVenue,externalIds"
    paper_details = make_single_api_request(f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}", {'fields': fields})
    time.sleep(1.0)
    if not paper_details: return None

    # --- NEW: Relevance Filter ---
    if not is_nlp_related_enhanced(paper_details):
        logging.warning(f"REJECTED: Paper {paper_id} ('{paper_details.get('title')}') does not meet the NLP relevance criteria.")
        return None

    logging.info(f"PASSED: Paper {paper_id} meets relevance criteria. Fetching connections.")
    
    # Fetch all references and citations
    references_raw = fetch_all_connections(paper_id, 'references')
    citations_raw = fetch_all_connections(paper_id, 'citations')
    
    # Process the data into the final format for the DataFrame
    paper_data = {
        'paperId': paper_details.get('paperId'),
        'title': paper_details.get('title'),
        'authors': json.dumps([a['name'] for a in paper_details.get('authors', [])]),
        'year': paper_details.get('year'),
        'citationCount': paper_details.get('citationCount'),
        's2FieldsOfStudy': json.dumps([f['category'] for f in paper_details.get('s2FieldsOfStudy', [])]),
        'publicationVenueName': paper_details.get('publicationVenue', {}).get('name'),
        'externalIdsACL': paper_details.get('externalIds', {}).get('ACL'),
        'externalIdsArXiv': paper_details.get('externalIds', {}).get('ArXiv'),
        'references_data': json.dumps(process_connections(references_raw, 'references')),
        'citations_data': json.dumps(process_connections(citations_raw, 'citations'))
    }
    return paper_data

def calculate_edge_weight(intents, is_influential, config):
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
    # Add nodes first
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Adding nodes"):
        node_id = row['paperId']
        if pd.isna(node_id): continue
        year_val, cit_count_val = row.get('year'), row.get('citationCount')
        attributes = {
            'title': str(row.get('title', '')), 'year': int(year_val) if pd.notna(year_val) else 0,
            'authors': str(row.get('authors', '')), 'citationCount': int(cit_count_val) if pd.notna(cit_count_val) else 0,
            's2FieldsOfStudy': str(row.get('s2FieldsOfStudy', '')), 'publicationVenueName': str(row.get('publicationVenueName', ''))
        }
        G.add_node(node_id, **attributes)
    # Add edges
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Adding edges"):
        source_node_id = row['paperId']
        if pd.isna(source_node_id): continue
        try:
            references = json.loads(row['references_data']) if row['references_data'] and pd.notna(row['references_data']) else []
        except (json.JSONDecodeError, TypeError): references = []
        for ref in references:
            target_node_id = ref.get('paperId')
            if target_node_id and G.has_node(target_node_id):
                weight = calculate_edge_weight(ref.get('intents', []), ref.get('isInfluential', False), weight_config)
                G.add_edge(source_node_id, target_node_id, weight=weight, intents=','.join(ref.get('intents', [])), isInfluential=ref.get('isInfluential', False))
    return G

# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Attempting to add paper '{PAPER_ID_TO_ADD}' to dataset '{INPUT_OUTPUT_CSV_FILE}'.")
    
    # 1. Load existing data
    try:
        df = pd.read_csv(INPUT_OUTPUT_CSV_FILE)
    except FileNotFoundError:
        logging.error(f"File not found: {INPUT_OUTPUT_CSV_FILE}. Cannot proceed.")
        df = pd.DataFrame() 

    # 2. Check if paper already exists
    if not df.empty and PAPER_ID_TO_ADD in df['paperId'].values:
        logging.warning(f"Paper {PAPER_ID_TO_ADD} is already in the dataset. No action taken.")
    else:
        # 3. Fetch data for the new paper
        new_paper_data = fetch_single_paper_data(PAPER_ID_TO_ADD)
        
        if new_paper_data:
            # 4. Append new paper to DataFrame
            new_row_df = pd.DataFrame([new_paper_data])
            df = pd.concat([df, new_row_df], ignore_index=True)
            
            # 5. Save the updated DataFrame back to the CSV
            try:
                df.to_csv(INPUT_OUTPUT_CSV_FILE, index=False)
                logging.info(f"Successfully added new paper and saved updated data to '{INPUT_OUTPUT_CSV_FILE}'. Total papers: {len(df)}")
            except Exception as e:
                logging.error(f"Failed to save updated CSV file: {e}")

    # 6. Re-build the entire graph from the (potentially updated) DataFrame
    if not df.empty:
        final_graph = build_graph_from_df(df, WEIGHT_CONFIG)
        logging.info(f"Graph rebuilt successfully with {final_graph.number_of_nodes()} nodes and {final_graph.number_of_edges()} edges.")
        
        # 7. Save the new graph
        try:
            nx.write_graphml(final_graph, OUTPUT_GRAPH_FILE)
            logging.info(f"Successfully saved updated graph to '{OUTPUT_GRAPH_FILE}'.")
        except Exception as e:
            logging.error(f"Failed to save the new graph file: {e}")
    else:
        logging.warning("DataFrame is empty, no graph was built or saved.")