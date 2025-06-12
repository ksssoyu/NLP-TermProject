import requests
import time
import csv
import json
import logging
import datetime
# This script requires the python-dateutil library.
# You can install it with: pip install python-dateutil
from dateutil.relativedelta import relativedelta

# --- Configuration ---
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx"
OUTPUT_CSV_FILE = "nlp_papers_2425_top_venues.csv"

# --- Top Venue List (from your is_nlp_related_enhanced filter) ---
# Only papers from these venues will be collected.
TOP_NLP_VENUES = [
    'acl', 'emnlp', 'naacl', 'coling', 'eacl',
    'transactions of the association for computational linguistics',
    'computational linguistics',
    'lrec',
    'neurips', 'icml', 'iclr', 'aaai', 'ijcai'
]

# --- Adjustable Search Parameters ---
SEARCH_QUERY = "natural language processing"
MAX_PAPERS_TO_COLLECT = 2000

# --- ADJUSTABLE TIME WINDOW ---
TIME_WINDOW_MONTHS = 6
YEAR_RANGE_START = 2024
YEAR_RANGE_END = 2025
MIN_CITATION_COUNT = 1
MAX_CITATION_COUNT = 9999

# --- API Settings ---
API_DELAY_SECONDS = 1.0
API_SEARCH_LIMIT = 100

# --- Fields to Request ---
REQUESTED_FIELDS = [
    'paperId', 'title', 'authors', 'year', 'citationCount',
    's2FieldsOfStudy', 'publicationVenue', 'externalIds', 'embedding'
]

# --- Logging Configuration ---
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

def search_and_fetch_by_custom_window():
    """
    Iterates through custom time windows to collect papers, filtering by top venues.
    """
    logging.info(f"Starting iterative search for top venue papers with a {TIME_WINDOW_MONTHS}-month window...")
    headers = {'X-API-KEY': API_KEY} if API_KEY and API_KEY != "YOUR_API_KEY_HERE" else {}
    if not headers:
        logging.warning("API_KEY is not set. Requests will be rate-limited.")

    collected_papers = []
    fields_param = ",".join(REQUESTED_FIELDS)
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    current_end_date = datetime.date(YEAR_RANGE_END, 12, 31)
    process_start_date = datetime.date(YEAR_RANGE_START, 1, 1)

    while current_end_date >= process_start_date:
        if len(collected_papers) >= MAX_PAPERS_TO_COLLECT:
            logging.info("Maximum paper collection limit reached. Halting search.")
            break

        current_start_date = current_end_date - relativedelta(months=TIME_WINDOW_MONTHS) + relativedelta(days=1)
        if current_start_date < process_start_date:
            current_start_date = process_start_date

        start_str = current_start_date.strftime('%Y-%m-%d')
        end_str = current_end_date.strftime('%Y-%m-%d')
        date_filter = f"{start_str}:{end_str}"
        
        logging.info(f"--- Querying for papers published between {start_str} and {end_str} ---")
        offset = 0
        papers_in_this_window = 0

        while papers_in_this_window < 2000:
            params = { 'query': SEARCH_QUERY, 'publicationDateOrYear': date_filter, 'fields': fields_param, 'offset': offset, 'limit': API_SEARCH_LIMIT }
            response_data = make_api_request(base_url, params, headers)

            if response_data == "retry": continue
            if not response_data or 'data' not in response_data: break

            papers_batch = response_data.get('data', [])
            if not papers_batch: break

            for paper in papers_batch:
                if papers_in_this_window >= 1000: break
                
                # --- NEW: VENUE FILTERING STEP ---
                pub_venue = paper.get('publicationVenue')
                if not pub_venue:
                    continue # Skip paper if it has no venue information
                
                venue_name = str(pub_venue.get('name', '')).lower()
                
                # Check if the venue name contains any of the top venue keywords
                if not any(top_venue in venue_name for top_venue in TOP_NLP_VENUES):
                    continue # Skip paper if it's not in a top venue

                # --- Existing Citation Filter ---
                citation_count = paper.get('citationCount')
                if citation_count is None or citation_count < MIN_CITATION_COUNT: continue
                if citation_count > MAX_CITATION_COUNT: continue
                
                # --- Add paper to list if it passes all filters ---
                formatted_paper = {
                    'paperId': paper.get('paperId'),
                    'title': paper.get('title'),
                    'authors': json.dumps([author.get('name') for author in paper.get('authors', []) if author.get('name')]),
                    'year': paper.get('year'),
                    'citationCount': citation_count,
                    's2FieldsOfStudy': json.dumps([fos.get('category') for fos in paper.get('s2FieldsOfStudy', []) if fos.get('category')]),
                    'publicationVenueName': pub_venue.get('name'),
                    'externalIdsACL': paper.get('externalIds', {}).get('ACL'),
                    'externalIdsArXiv': paper.get('externalIds', {}).get('ArXiv'),
                    'embedding': json.dumps(paper.get('embedding', {}).get('vector')) if paper.get('embedding') else None
                }
                collected_papers.append(formatted_paper)
                papers_in_this_window += 1

            logging.info(f"Collected {len(collected_papers)} papers in total. ({papers_in_this_window} from top venues in this window)")
            
            if 'next' in response_data and response_data['next'] < 1000:
                offset = response_data['next']
            else:
                break
        
        current_end_date = current_start_date - relativedelta(days=1)

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
    papers = search_and_fetch_by_custom_window()
    if papers:
        save_to_csv(papers, OUTPUT_CSV_FILE)
    logging.info("Script finished.")