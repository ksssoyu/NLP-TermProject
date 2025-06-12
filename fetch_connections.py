import pandas as pd
import requests
import time
import json
import logging
from tqdm import tqdm

# --- Configuration ---
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx"
# Use the merged file as input
INPUT_CSV_FILE = "merged_nlp_papers_without_embeddings.csv"
# Define the final output file
OUTPUT_CSV_FILE = "final_nlp_papers.csv"

# --- API Settings ---
API_DELAY_SECONDS = 1.0
PAGINATED_ENDPOINT_LIMIT = 1000

# --- Progress Saving ---
SAVE_PROGRESS_INTERVAL = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_all_connections(paper_id, connection_type, headers):
    """Fetches all citations or references for a paper using pagination."""
    # This function remains the same as before
    all_connections = []
    offset = 0
    fields = "isInfluential,intents," + \
             ("citingPaper.paperId" if connection_type == "citations" else "citedPaper.paperId")
    
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{connection_type}"

    while True:
        params = {'fields': fields, 'limit': PAGINATED_ENDPOINT_LIMIT, 'offset': offset}
        try:
            time.sleep(API_DELAY_SECONDS)
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error for {paper_id} ({connection_type}): {e}")
            break

        data = response_json.get('data', [])
        if not data:
            break
        
        all_connections.extend(data)
        
        if 'next' in response_json:
            offset = response_json['next']
        else:
            break
            
    return all_connections

def process_connections(raw_connections, connection_type):
    """Processes the raw connection data into a structured list."""
    # This function remains the same as before
    processed_list = []
    key = "citingPaper" if connection_type == "citations" else "citedPaper"
    for item in raw_connections:
        paper_info = item.get(key, {})
        if paper_id := paper_info.get('paperId'):
            processed_list.append({
                'paperId': paper_id,
                'isInfluential': item.get('isInfluential', False),
                'intents': item.get('intents', [])
            })
    return processed_list

def enrich_dataset(input_path, output_path):
    """Reads the base dataset and enriches it with connection data."""
    logging.info(f"Loading base dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        return

    # --- IMPORTANT: Ensure the columns for enrichment exist ---
    if 'references_data' not in df.columns:
        df['references_data'] = None
    if 'citations_data' not in df.columns:
        df['citations_data'] = None

    headers = {'X-API-KEY': API_KEY} if API_KEY and API_KEY != "YOUR_API_KEY_HERE" else {}
    if not headers:
        logging.warning("API_KEY is not set. Requests will be rate-limited.")

    logging.info(f"Starting to enrich {len(df)} papers. Will skip already enriched papers.")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Enriching Papers"):
        
        # --- THE CRUCIAL EFFICIENCY CHECK ---
        # If 'references_data' is not empty (i.e., it's not NaN/None),
        # it means the paper is from the original dataset and already enriched.
        # We can safely skip it.
        if pd.notna(row['references_data']):
            continue

        paper_id = row['paperId']
        logging.info(f"Fetching connection data for new paper: {paper_id}")
        
        # Fetch References
        raw_references = fetch_all_connections(paper_id, 'references', headers)
        processed_references = process_connections(raw_references, 'references')
        df.at[index, 'references_data'] = json.dumps(processed_references)

        # Fetch Citations
        raw_citations = fetch_all_connections(paper_id, 'citations', headers)
        processed_citations = process_connections(raw_citations, 'citations')
        df.at[index, 'citations_data'] = json.dumps(processed_citations)
        
        # Save progress
        if (index + 1) % SAVE_PROGRESS_INTERVAL == 0:
            logging.info(f"Saving progress at paper {index + 1}...")
            df.to_csv(output_path, index=False)

    logging.info("Enrichment complete. Saving final dataset...")
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved enriched dataset to '{output_path}'.")

if __name__ == "__main__":
    enrich_dataset(INPUT_CSV_FILE, OUTPUT_CSV_FILE)