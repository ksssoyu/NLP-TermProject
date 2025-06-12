import requests
import time
import csv
import json
from collections import deque
import logging

# --- Configuration ---
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx"
OUTPUT_CSV_FILE = "nlp_papers_continuous_bfs_final.csv"

# --- Adjustable Parameters ---
MIN_CITATION_COUNT = 5
MAX_PAPERS_TO_COLLECT = 12000
YEAR_RANGE_START = 2010
YEAR_RANGE_END = None
API_DELAY_SECONDS = 1.0

SEED_PAPER_IDS = [
    "df2b0e26d0599ce3e70df8a9da02e51594e0e992", # bert
    "204e3073870fae3d05bcbc2f6a8e263d9b72e776", # attention is all you need
    "9405cc0d6169988371b2755e573cc28650d14dfe", # gpt2
    "90abbc2cf38462b954ae1b772fac9532e2ccd8b0", # gpt3
    "f6b51c8753a871dc94ff32152c00c01e94f90f09", # word2vec
    "f37e1b62a767a307c046404ca96bc140b3e68cb5", # glove
    "cea967b59209c6be22829699f05b8b1ac4dc092d", # encoder-decoder
    "0b544dfe355a5070b60986319a3f51fb45d1348e", # encoder-decoder
    "fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5", # attention mechanism
    "2e9d221c206e9503ceb452302d68d10e293f2a10", # LSTM
    "d7da009f457917aa381619facfa5ffae9329a6e9", # bleu
    "6c2b28f9354f667cd5bd07afc0471d8334430da7", # Probabilistic Language Model
    "3febb2bed8865945e7fddc99efd791887bb7e14f", # ELMo
    "d766bffc357127e0dc86dd69561d5aeb520d6f4c", # instructGPT  
    "1b6e810ce0afd0dd093f789d2b2742d047e316d5", # chain of thought
    "094ff971d6a8b8ff870946c9b3ce5aa173617bfb", # PaLM
    "451d4a16e425ecbf38c4b1cca0dcf5d9bec8255c", # GLUE
    "a54b56af24bb4873ed0163b77df63b92bd018ddc"  # DistilBERT
]

# --- API Limits & New Connection Limit ---
PAGINATED_ENDPOINT_LIMIT = 100
CONNECTION_FETCH_LIMIT = 500

# --- NLP Relevance Constants ---
CORE_S2_NLP_FOS = ['natural language processing', 'computational linguistics']
TOP_NLP_VENUES = [
    'acl', 'emnlp', 'naacl', 'coling', 'eacl',
    'transactions of the association for computational linguistics', 'computational linguistics', 'lrec',
    'neurips', 'icml', 'iclr', 'aaai', 'ijcai'
]
BROADER_S2_NLP_FOS = ['artificial intelligence', 'linguistics', 'information retrieval', 'speech recognition', 'machine learning']
CS_FOS_CATEGORY = "computer science"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_api_request(url, params, headers, request_type="generic"):
    """Makes a single API request and handles common errors with detailed logging."""
    try:
        time.sleep(API_DELAY_SECONDS)
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        log_params = f"Params: {params}"
        if e.response.status_code == 404: logging.warning(f"{request_type.capitalize()} data not found (404) for URL {url}. {log_params}")
        elif e.response.status_code == 429:
            logging.warning(f"Rate limit (429) for {request_type} URL {url}. Pausing. {log_params}")
            time.sleep(10)
        else: logging.error(f"HTTP error {e.response.status_code} for {request_type} URL {url}. {log_params}. Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error for {request_type} URL {url}. Params: {params}. Error: {e}")
    return None

def fetch_and_process_connections(paper_id, connection_type, headers):
    """Fetches and processes detailed connection data up to the CONNECTION_FETCH_LIMIT."""
    all_items, processed_list = [], []
    offset = 0
    fields = f"intents,isInfluential,{'citingPaper.paperId' if connection_type == 'citations' else 'citedPaper.paperId'}"
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{connection_type}"
    key_for_connected_paper = "citingPaper" if connection_type == "citations" else "citedPaper"
    
    while True:
        if len(all_items) >= CONNECTION_FETCH_LIMIT: break
        
        params = {'fields': fields, 'limit': PAGINATED_ENDPOINT_LIMIT, 'offset': offset}
        response = make_api_request(base_url, params, headers, request_type=f"{connection_type} for {paper_id}")
        if not response: break
        
        data = response.get('data', [])
        if not data: break
        all_items.extend(data)
        
        if response.get('next') is not None: offset = response['next']
        else: break
            
    for item in all_items[:CONNECTION_FETCH_LIMIT]:
        paper_info = item.get(key_for_connected_paper, {})
        if conn_paper_id := paper_info.get('paperId'):
            processed_list.append({ 'paperId': conn_paper_id, 'isInfluential': item.get('isInfluential', False), 'intents': item.get('intents', []) })
    return processed_list

def is_nlp_related(api_response_data):
    """Updated NLP relevance check."""
    if not api_response_data: return False
    external_ids = api_response_data.get('externalIds', {})
    if external_ids.get('ACL') or external_ids.get('ArXiv'): return True
    
    pub_venue = api_response_data.get('publicationVenue')
    if pub_venue and any(tv in str(pub_venue.get('name', '')).lower() for tv in TOP_NLP_VENUES): return True
            
    s2_cats = [f.get('category', '').lower() for f in api_response_data.get('s2FieldsOfStudy', []) if f.get('category')]
    if CS_FOS_CATEGORY in s2_cats or any(cf in s2_cats for cf in CORE_S2_NLP_FOS) or any(bf in s2_cats for bf in BROADER_S2_NLP_FOS): return True
    return False

def fetch_papers_bfs():
    """Performs a continuous BFS, limiting connections added to the queue."""
    headers = {'X-API-KEY': API_KEY} if API_KEY else {}
    queue = deque(SEED_PAPER_IDS)
    visited_ids = set(SEED_PAPER_IDS)
    collected_papers = []

    fields_to_request = 'paperId,title,authors,year,citationCount,s2FieldsOfStudy,publicationVenue,externalIds,embedding,references.paperId,citations.paperId'
    
    while queue and len(collected_papers) < MAX_PAPERS_TO_COLLECT:
        paper_id = queue.popleft()
        logging.info(f"Processing paper: {paper_id} (Queue: {len(queue)}, Collected: {len(collected_papers)}/{MAX_PAPERS_TO_COLLECT})")
        
        paper_response = make_api_request(f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}", {'fields': fields_to_request}, headers, request_type=f"main paper {paper_id}")
        if not paper_response: continue

        # Apply Filters
        if not (paper_response.get('year') and paper_response.get('year') >= YEAR_RANGE_START and (YEAR_RANGE_END is None or paper_response.get('year') <= YEAR_RANGE_END)): continue
        if not (paper_response.get('citationCount') and paper_response.get('citationCount') >= MIN_CITATION_COUNT): continue
        if not is_nlp_related(paper_response): continue

        references_data = fetch_and_process_connections(paper_id, 'references', headers)
        citations_data = fetch_and_process_connections(paper_id, 'citations', headers)
        
        pub_venue_obj = paper_response.get('publicationVenue')
        venue_name = pub_venue_obj.get('name') if pub_venue_obj else None
        
        formatted_paper = {
            'paperId': paper_response.get('paperId'), 'title': paper_response.get('title'),
            'authors': json.dumps([a.get('name') for a in paper_response.get('authors', []) if a.get('name')]),
            'year': paper_response.get('year'), 'citationCount': paper_response.get('citationCount'),
            's2FieldsOfStudy': json.dumps([f.get('category') for f in paper_response.get('s2FieldsOfStudy', []) if f.get('category')]),
            'publicationVenueName': venue_name,
            'externalIdsACL': paper_response.get('externalIds', {}).get('ACL'),
            'externalIdsArXiv': paper_response.get('externalIds', {}).get('ArXiv'),
            'embedding': json.dumps(paper_response.get('embedding', {}).get('vector')) if paper_response.get('embedding') else None,
            'references_data': json.dumps(references_data),
            'citations_data': json.dumps(citations_data)
        }
        collected_papers.append(formatted_paper)
        
        # --- FIX: Safely get and slice the list of connections ---
        references_list = paper_response.get('references')
        if references_list: # Check if the list is not None
            for ref in references_list[:CONNECTION_FETCH_LIMIT]:
                if ref and (ref_id := ref.get('paperId')) and ref_id not in visited_ids:
                    visited_ids.add(ref_id); queue.append(ref_id)
        
        citations_list = paper_response.get('citations')
        if citations_list: # Check if the list is not None
            for cit in citations_list[:CONNECTION_FETCH_LIMIT]:
                if cit and (cit_id := cit.get('paperId')) and cit_id not in visited_ids:
                    visited_ids.add(cit_id); queue.append(cit_id)

    return collected_papers

def save_to_csv(data_list, filename):
    """Saves the collected paper data to a CSV file."""
    if not data_list: logging.info("No data to save."); return
    logging.info(f"Saving {len(data_list)} papers to {filename}...")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_list[0].keys())
            writer.writeheader()
            writer.writerows(data_list)
        logging.info(f"Successfully saved data to {filename}")
    except (IOError, IndexError) as e: logging.error(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    logging.info("Starting Continuous BFS Paper Collection (v6 - Final)...")
    papers = fetch_papers_bfs()
    if papers: save_to_csv(papers, OUTPUT_CSV_FILE)
    logging.info("Script finished.")