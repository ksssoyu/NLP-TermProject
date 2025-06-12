import pandas as pd
import requests
import time
import json
import logging
from tqdm import tqdm

# --- Configuration ---
# The path to your existing dataset CSV file.
INPUT_CSV_FILE = "nlp_papers_dataset_v6_with_isInfluential.csv"
# The path for the new CSV file that will include the embedding vectors.
OUTPUT_CSV_FILE = "papers_with_embeddings.csv"

# --- API Configuration ---
# Your Semantic Scholar API Key (optional but highly recommended for long runs)
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx"
API_DELAY_SECONDS = 1.0
# The specific field we want to fetch
FIELDS_TO_FETCH = "embedding"

# --- Progress Saving ---
# Save the file after every N papers to avoid data loss on long runs.
SAVE_PROGRESS_INTERVAL = 200

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Functions ---

def make_embedding_api_request(paper_id):
    """Makes a single API request to fetch the embedding for one paper."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {'fields': FIELDS_TO_FETCH}
    headers = {}
    if API_KEY and API_KEY != "YOUR_SEMANTIC_SCHOLAR_API_KEY":
        headers['X-API-KEY'] = API_KEY
    
    try:
        time.sleep(API_DELAY_SECONDS)
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # It's common for some papers not to have embeddings, so a 404 is not a critical error.
        if e.response.status_code == 404:
            logging.warning(f"Paper {paper_id} not found or has no embedding (404).")
        else:
            logging.error(f"HTTP error for paper {paper_id}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred for paper {paper_id}: {e}")
    return None

def fetch_embeddings_for_dataset(csv_path):
    """
    Reads a CSV of paper IDs, fetches the embedding for each, and saves to a new CSV.
    """
    logging.info(f"Loading data from {csv_path} to fetch embeddings...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}. Cannot proceed.")
        return

    # Add a new column for the embedding vector, initialized to None
    df['embedding'] = None

    logging.info(f"Starting to fetch embeddings for {len(df)} papers...")
    
    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching Embeddings"):
        paper_id = row['paperId']
        
        # Skip if embedding has already been fetched (in case of script restart)
        if pd.notna(row['embedding']):
            continue

        api_response = make_embedding_api_request(paper_id)
        
        if api_response and 'embedding' in api_response and api_response['embedding']:
            # The API returns an object {'model': '...', 'vector': [...]}. We want the vector.
            embedding_vector = api_response['embedding'].get('vector')
            if embedding_vector:
                # Store the vector as a JSON string in the DataFrame cell
                df.at[index, 'embedding'] = json.dumps(embedding_vector)

        # Save progress periodically
        if (index + 1) % SAVE_PROGRESS_INTERVAL == 0:
            try:
                logging.info(f"Saving progress at paper {index + 1}/{len(df)}...")
                df.to_csv(OUTPUT_CSV_FILE, index=False)
            except Exception as e:
                logging.error(f"Failed to save progress to {OUTPUT_CSV_FILE}: {e}")
    
    # Final save at the end
    try:
        logging.info("Embedding fetch complete. Saving final dataset...")
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        logging.info(f"Successfully saved dataset with embeddings to '{OUTPUT_CSV_FILE}'.")
    except Exception as e:
        logging.error(f"Failed to save final CSV file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    fetch_embeddings_for_dataset(INPUT_CSV_FILE)
