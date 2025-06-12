import requests
import time
import csv
import json
import logging

# --- Configuration ---
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx"
OUTPUT_CSV_FILE = "newly_fetched_2425_nlp_papers.csv"

# --- Adjustable Search Parameters ---
SEARCH_QUERY = "natural language processing"
MAX_PAPERS_TO_COLLECT = 1000  # Target a larger number of papers
YEAR_RANGE_START = 2024 # Focus on more recent years
YEAR_RANGE_END = 2026
MIN_CITATION_COUNT = 1 # It's okay to include recent papers with few citations
MAX_CITATION_COUNT = 200 # New parameter to exclude "superstar" papers

# --- API Settings ---
API_DELAY_SECONDS = 1.0
API_SEARCH_LIMIT = 100

# --- Fields to Request from the Search API ---
REQUESTED_FIELDS = [
    'paperId', 'title', 'authors', 'year', 'citationCount',
    's2FieldsOfStudy', 'publicationVenue', 'externalIds', 'embedding'
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_api_request(url, params, headers):
    """Makes a single API request and handles common errors."""
    try:
        time.sleep(API_DELAY_SECONDS)
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logging.warning("Rate limit exceeded. Waiting for 60 seconds.")
            time.sleep(60)
            return "retry"
        logging.error(f"HTTP Error: {e.response.status_code} for URL {url} with params {params}")
        logging.error(f"Response text: {e.response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
    return None

def search_and_fetch_papers():
    """
    Uses the /paper/search endpoint to find papers matching a query,
    applies filters, and collects them until the max count is reached.
    """
    logging.info(f"Starting paper search for query: '{SEARCH_QUERY}'")
    headers = {'X-API-KEY': API_KEY} if API_KEY and API_KEY != "YOUR_API_KEY_HERE" else {}
    if not headers:
        logging.warning("API_KEY is not set. Requests will be rate-limited.")

    collected_papers = []
    offset = 0
    
    fields_param = ",".join(REQUESTED_FIELDS)
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    while len(collected_papers) < MAX_PAPERS_TO_COLLECT:
        params = {
            'query': SEARCH_QUERY,
            'fields': fields_param,
            'offset': offset,
            'limit': API_SEARCH_LIMIT
        }
        
        logging.info(f"Requesting papers from offset {offset}...")
        response_data = make_api_request(base_url, params, headers)

        if response_data == "retry":
            continue

        if not response_data or 'data' not in response_data:
            logging.error("Failed to fetch data or received an invalid response.")
            break

        papers_batch = response_data.get('data', [])
        if not papers_batch:
            logging.info("No more papers found in search results.")
            break

        for paper in papers_batch:
            if len(collected_papers) >= MAX_PAPERS_TO_COLLECT:
                break
            
            # --- Apply Filters ---
            year = paper.get('year')
            citation_count = paper.get('citationCount')

            if year is None or year < YEAR_RANGE_START:
                continue
            if YEAR_RANGE_END is not None and year > YEAR_RANGE_END:
                continue
            if citation_count is None or citation_count < MIN_CITATION_COUNT:
                continue
            if citation_count > MAX_CITATION_COUNT: # Apply the new max citation filter
                continue
            
            # --- Format Data for CSV ---
            formatted_paper = {
                'paperId': paper.get('paperId'),
                'title': paper.get('title'),
                'authors': json.dumps([author.get('name') for author in paper.get('authors', []) if author.get('name')]),
                'year': year,
                'citationCount': citation_count,
                's2FieldsOfStudy': json.dumps([fos.get('category') for fos in paper.get('s2FieldsOfStudy', []) if fos.get('category')]),
                'publicationVenueName': paper.get('publicationVenue', {}).get('name') if paper.get('publicationVenue') else None,
                'externalIdsACL': paper.get('externalIds', {}).get('ACL'),
                'externalIdsArXiv': paper.get('externalIds', {}).get('ArXiv'),
                'embedding': json.dumps(paper.get('embedding', {}).get('vector')) if paper.get('embedding') else None
            }
            collected_papers.append(formatted_paper)

        logging.info(f"Collected {len(collected_papers)}/{MAX_PAPERS_TO_COLLECT} papers.")
        
        if 'next' in response_data and response_data['next'] < response_data.get('total', 0):
            offset = response_data['next']
        else:
            logging.info("Reached the end of the search results.")
            break

    return collected_papers

def save_to_csv(data_list, filename):
    """Saves the collected paper data to a CSV file."""
    if not data_list:
        logging.info("No data to save.")
        return

    logging.info(f"Saving {len(data_list)} papers to {filename}...")
    headers = data_list[0].keys()
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_list)
        logging.info(f"Successfully saved data to {filename}")
    except IOError as e:
        logging.error(f"Failed to write to CSV file {filename}: {e}")

if __name__ == "__main__":
    papers = search_and_fetch_papers()
    if papers:
        save_to_csv(papers, OUTPUT_CSV_FILE)
    logging.info("Script finished.")