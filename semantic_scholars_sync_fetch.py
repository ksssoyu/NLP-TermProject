import requests
import time
import csv
import json
from collections import deque
import logging

# --- Configuration ---
API_KEY = "KRzThuhqxK42MZRgsnQJR5NnLLO17hKx8flTuDnx" # <<< YOUR API KEY HERE

# --- Adjustable Parameters ---
MIN_CITATION_COUNT = 0  
MAX_PAPERS_TO_COLLECT = 10000
YEAR_RANGE_START = 2010
YEAR_RANGE_END = None 
API_DELAY_SECONDS = 1.0
OUTPUT_CSV_FILE = "nlp_papers_dataset_v6_with_isInfluential.csv" # Changed output file name

# --- API Limits ---
MAIN_ENDPOINT_CONNECTION_BATCH_LIMIT = 1000 
PAGINATED_ENDPOINT_LIMIT = 1000

REQUESTED_METADATA_FIELDS = [  
    'paperId', 'title', 'authors', 'year', 'citationCount',
    's2FieldsOfStudy', 'publicationVenueName', 'externalIdsACL', 'externalIdsArXiv',
    'references_data', # Will now include 'isInfluential'
    'citations_data'   # Will now include 'isInfluential'
]

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
    "3febb2bed8865945e7fddc99efd791887bb7e14f"  # ELMo
]
CORE_S2_NLP_FOS = ['natural language processing', 'computational linguistics']
TOP_NLP_VENUES = [
    'acl', 'emnlp', 'naacl', 'coling', 'eacl',
    'transactions of the association for computational linguistics', 'computational linguistics', 'lrec',
    'neurips', 'icml', 'iclr', 'aaai', 'ijcai'
]
BROADER_S2_NLP_FOS = ['artificial intelligence', 'linguistics', 'information retrieval', 'speech recognition', 'machine learning']
CS_FOS_CATEGORY = "computer science"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_api_fields_string(requested_csv_fields):
    api_fields_set = {
        'paperId', 'title', 'year', 'citationCount',
        'externalIds', 'publicationVenue', 's2FieldsOfStudy',
        'references.paperId', 
        'citations.paperId'   
    }
    if 'authors' in requested_csv_fields: 
        api_fields_set.add('authors.name')
    return ",".join(sorted(list(api_fields_set)))

def make_single_api_request(url, params, api_key_to_use, delay_seconds_to_use, request_type="generic"):
    headers = {}
    if api_key_to_use and api_key_to_use != "YOUR_SEMANTIC_SCHOLAR_API_KEY":
        headers['X-API-KEY'] = api_key_to_use
    else:
        logging.warning(f"API_KEY not set for {request_type} request to {url}.")
    try:
        time.sleep(delay_seconds_to_use)
        logging.debug(f"Making API {request_type} request: URL={url}, Params={params}")
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400: logging.error(f"HTTP Bad Request (400) for {request_type} URL {url}. Params: {params}. Response: {e.response.text}")
        elif e.response.status_code == 404: logging.warning(f"{request_type.capitalize()} data not found (404) for URL {url}.")
        elif e.response.status_code == 403: logging.error(f"Forbidden (403) for {request_type} URL {url}. Check API key.")
        elif e.response.status_code == 429:
            logging.warning(f"Rate limit (429) for {request_type} URL {url}. Pausing longer.")
            time.sleep(delay_seconds_to_use * 10) 
        else: logging.error(f"HTTP error {e.response.status_code} for {request_type} URL {url}: {e.response.text}")
    except requests.exceptions.RequestException as e: logging.error(f"Request error for {request_type} URL {url}: {e}")
    return None

def fetch_all_connections_with_intents_and_influence(paper_id_of_interest, connection_type, api_key_to_use, delay_seconds_to_use): # Renamed function
    """
    Fetches all connections (citations or references) for a given paper using pagination
    from the dedicated /references or /citations endpoints, requesting 'intents' and 'isInfluential'.
    """
    all_connection_items = []
    offset = 0
    # Request 'intents', 'isInfluential', and connected paper's ID.
    fields_for_connection = "intents,isInfluential," + \
                            ("citingPaper.paperId" if connection_type == "citations" else "citedPaper.paperId")

    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id_of_interest}/{connection_type}"
    logging.info(f"Fetching all {connection_type} with intents & influence for {paper_id_of_interest} via pagination from {base_url}...")

    while True:
        params = {'fields': fields_for_connection, 'limit': PAGINATED_ENDPOINT_LIMIT, 'offset': offset}
        response_json = make_single_api_request(base_url, params, api_key_to_use, delay_seconds_to_use, request_type=f"{connection_type}_extended") # Updated request_type

        if not response_json: 
            logging.error(f"Failed to fetch {connection_type} extended page for {paper_id_of_interest} at offset {offset}.")
            break 
        
        current_batch = response_json.get('data', [])
        if not current_batch: break
        
        all_connection_items.extend(current_batch)
        
        next_offset_val = response_json.get('next')
        if next_offset_val is not None and len(current_batch) == PAGINATED_ENDPOINT_LIMIT :
            offset = next_offset_val
        else: break
            
    logging.info(f"Fetched a total of {len(all_connection_items)} {connection_type} with intents & influence for {paper_id_of_interest}.")
    return all_connection_items

def adapt_and_process_connections_for_intents_and_influence(raw_connections_list, connection_type): # Renamed function
    """
    Adapts raw connection data (which includes 'intents' and 'isInfluential') 
    and processes it for CSV output.
    """
    adapted_list = []
    key_for_connected_paper = "citingPaper" if connection_type == "citations" else "citedPaper"

    for raw_item in raw_connections_list: 
        connected_paper_info = raw_item.get(key_for_connected_paper, {})
        connected_paper_id = connected_paper_info.get('paperId')
        direct_intents_list = raw_item.get('intents', []) 
        is_influential_flag = raw_item.get('isInfluential', False) # Default to False if not present

        if connected_paper_id: 
            adapted_list.append({
                'paperId': connected_paper_id, 
                'intents_list': direct_intents_list,
                'isInfluential': is_influential_flag # Add the influential flag
            })
        else:
            logging.warning(f"Missing '{key_for_connected_paper}.paperId' in raw {connection_type} item: {raw_item}")
            
    return process_citation_data_with_influence(adapted_list) # Call new processing function

def is_nlp_related_enhanced(api_response_data):
    # (This function remains the same as in the previous version with CS check)
    if not api_response_data: return False
    paper_id_for_log = api_response_data.get('paperId', 'UNKNOWN_ID'); title_for_log = api_response_data.get('title', 'NO_TITLE')[:70]
    external_ids_log = api_response_data.get('externalIds', {}); pub_venue_data_log = api_response_data.get('publicationVenue') 
    venue_name_log = "NO_VENUE_DATA"
    if pub_venue_data_log and isinstance(pub_venue_data_log, dict): venue_name_log = pub_venue_data_log.get('name', 'NO_VENUE_NAME').lower()
    s2_fields_log = api_response_data.get('s2FieldsOfStudy', []); s2_categories_log = [field.get('category', '').lower() for field in s2_fields_log if field.get('category')]
    if external_ids_log.get('ACL'): logging.debug(f"  Relevance Check (Paper: {paper_id_for_log} - {title_for_log}): True (ACL ID found)"); return True
    if pub_venue_data_log and isinstance(pub_venue_data_log, dict):
        for top_venue in TOP_NLP_VENUES:
            if top_venue in venue_name_log: logging.debug(f"  Relevance Check (Paper: {paper_id_for_log} - {title_for_log}): True (Top Venue: '{venue_name_log}')"); return True
    if CS_FOS_CATEGORY in s2_categories_log: logging.debug(f"  Relevance Check (Paper: {paper_id_for_log} - {title_for_log}): True (S2 FOS includes '{CS_FOS_CATEGORY}')"); return True
    for core_fos in CORE_S2_NLP_FOS:
        if core_fos in s2_categories_log: logging.debug(f"  Relevance Check (Paper: {paper_id_for_log} - {title_for_log}): True (Core S2 FOS: '{core_fos}')"); return True
    has_arxiv_id_log = bool(external_ids_log.get('ArXiv')); nlp_signal_count_log = 0; found_broader_fos_log = []
    for broader_fos in BROADER_S2_NLP_FOS:
        if broader_fos in s2_categories_log: nlp_signal_count_log +=1; found_broader_fos_log.append(broader_fos)
    if (has_arxiv_id_log and nlp_signal_count_log > 0): logging.debug(f"  Relevance Check (Paper: {paper_id_for_log} - {title_for_log}): True (ArXiv + Broader FOS: {found_broader_fos_log})"); return True
    if (nlp_signal_count_log > 0): logging.debug(f"  Relevance Check (Paper: {paper_id_for_log} - {title_for_log}): True (Broader FOS: {found_broader_fos_log})"); return True
    logging.info(f"  Relevance Check (Paper: {paper_id_for_log} - {title_for_log}): FALSE. Details:\n    External IDs: {external_ids_log}\n    Venue: {venue_name_log}\n    S2 Categories: {s2_categories_log}"); return False

def process_citation_data_with_influence(adapted_items_list): # Renamed and updated
    """
    Processes a list of adapted items, where each item has 'paperId', 
    'intents_list', and 'isInfluential'.
    """
    processed_list = []
    if not adapted_items_list: return processed_list
    for item in adapted_items_list:
        item_id = item.get('paperId') 
        if not item_id: continue
        
        intents_set = set()
        direct_intents_list = item.get('intents_list', [])
        is_influential_val = item.get('isInfluential', False) # Get the influential flag

        if direct_intents_list:
            for intent_str in direct_intents_list:
                if intent_str and isinstance(intent_str, str): intents_set.add(intent_str.strip().lower())
                elif intent_str: logging.warning(f"Non-string intent for paper {item_id}: {intent_str} (type: {type(intent_str)})")
        
        processed_list.append({
            'paperId': item_id, 
            'intents': sorted(list(intents_set)),
            'isInfluential': is_influential_val # Store the flag
        })
    return processed_list

def process_paper_data(api_response_main_call, requested_csv_fields, current_api_key, current_api_delay):
    if not api_response_main_call: return None
    output_data = {}
    paper_id_of_current_paper = api_response_main_call.get('paperId')
    output_data['paperId'] = paper_id_of_current_paper
    # Standard field extraction
    if 'title' in requested_csv_fields: output_data['title'] = api_response_main_call.get('title')
    if 'authors' in requested_csv_fields:
        authors_list = api_response_main_call.get('authors', [])
        output_data['authors'] = [author['name'] for author in authors_list if author.get('name')] if authors_list else []
    if 'year' in requested_csv_fields: output_data['year'] = api_response_main_call.get('year')
    if 'citationCount' in requested_csv_fields: output_data['citationCount'] = api_response_main_call.get('citationCount')
    if 's2FieldsOfStudy' in requested_csv_fields:
        s2fos_list_from_api = api_response_main_call.get('s2FieldsOfStudy', [])
        output_data['s2FieldsOfStudy'] = sorted(list(set(fos.get('category') for fos in s2fos_list_from_api if fos.get('category')))) if s2fos_list_from_api else []
    if 'publicationVenueName' in requested_csv_fields:
        pub_venue_obj_from_api = api_response_main_call.get('publicationVenue')
        output_data['publicationVenueName'] = pub_venue_obj_from_api.get('name') if isinstance(pub_venue_obj_from_api, dict) else None
    if 'externalIdsACL' in requested_csv_fields: output_data['externalIdsACL'] = api_response_main_call.get('externalIds', {}).get('ACL')
    if 'externalIdsArXiv' in requested_csv_fields: output_data['externalIdsArXiv'] = api_response_main_call.get('externalIds', {}).get('ArXiv')

    if 'references_data' in requested_csv_fields:
        logging.info(f"Fetching full references with intents & influence for collected paper {paper_id_of_current_paper}...")
        all_raw_references = fetch_all_connections_with_intents_and_influence(paper_id_of_current_paper, 'references', current_api_key, current_api_delay)
        output_data['references_data'] = adapt_and_process_connections_for_intents_and_influence(all_raw_references, 'references')
    else: output_data['references_data'] = [] 
    if 'citations_data' in requested_csv_fields:
        logging.info(f"Fetching full citations with intents & influence for collected paper {paper_id_of_current_paper}...")
        all_raw_citations = fetch_all_connections_with_intents_and_influence(paper_id_of_current_paper, 'citations', current_api_key, current_api_delay)
        output_data['citations_data'] = adapt_and_process_connections_for_intents_and_influence(all_raw_citations, 'citations')
    else: output_data['citations_data'] = []
    final_output_for_csv = {field: output_data.get(field) for field in requested_csv_fields}
    return final_output_for_csv

def fetch_papers_bfs():
    # (This function's core BFS logic remains the same as the previous version)
    if not API_KEY or API_KEY == "YOUR_SEMANTIC_SCHOLAR_API_KEY": logging.error("CRITICAL: API_KEY is missing.")
    if not SEED_PAPER_IDS: logging.warning("SEED_PAPER_IDS is empty."); return []
    queue = deque(SEED_PAPER_IDS); visited_for_processing = set(SEED_PAPER_IDS)
    collected_paper_ids_set = set(); collected_papers_data = []
    main_paper_api_fields_param = build_api_fields_string(REQUESTED_METADATA_FIELDS)
    logging.info(f"Requesting main paper API fields (for filtering & BFS): {main_paper_api_fields_param}")
    papers_processed_count = 0
    while queue and len(collected_papers_data) < MAX_PAPERS_TO_COLLECT:
        current_paper_id = queue.popleft(); papers_processed_count += 1
        logging.info(f"Processing paper #{papers_processed_count}: {current_paper_id} (Queue: {len(queue)}, Collected: {len(collected_papers_data)}/{MAX_PAPERS_TO_COLLECT})")
        main_paper_api_response = make_single_api_request(
            f"https://api.semanticscholar.org/graph/v1/paper/{current_paper_id}",
            {'fields': main_paper_api_fields_param}, API_KEY, API_DELAY_SECONDS, request_type="main paper detail"
        )
        if not main_paper_api_response: logging.debug(f"  Skipping {current_paper_id} due to main API fetch failure."); continue
        year = main_paper_api_response.get('year'); citation_count_val = main_paper_api_response.get('citationCount')
        passes_filters = True; reason_for_pre_filter = []
        if year is None or year < YEAR_RANGE_START: passes_filters = False; reason_for_pre_filter.append(f"Year ({year}) < {YEAR_RANGE_START}")
        if YEAR_RANGE_END is not None and (year is None or year > YEAR_RANGE_END): passes_filters = False; reason_for_pre_filter.append(f"Year ({year}) > {YEAR_RANGE_END}")
        if citation_count_val is None or citation_count_val < MIN_CITATION_COUNT: passes_filters = False; reason_for_pre_filter.append(f"Citations ({citation_count_val}) < {MIN_CITATION_COUNT}")
        if not passes_filters: logging.info(f"  Paper {current_paper_id} (Title: {main_paper_api_response.get('title', 'N/A')[:60]}...) pre-filtered out: {'; '.join(reason_for_pre_filter)}.")
        elif not is_nlp_related_enhanced(main_paper_api_response): passes_filters = False
        if passes_filters and current_paper_id not in collected_paper_ids_set:
            paper_details_for_csv = process_paper_data(main_paper_api_response, REQUESTED_METADATA_FIELDS, API_KEY, API_DELAY_SECONDS)
            if paper_details_for_csv:
                collected_papers_data.append(paper_details_for_csv); collected_paper_ids_set.add(current_paper_id)
                logging.info(f"  Stored paper {current_paper_id} (Title: {main_paper_api_response.get('title', 'N/A')[:60]}...).")
        elif not passes_filters and current_paper_id not in collected_paper_ids_set and not reason_for_pre_filter : pass
        elif current_paper_id in collected_paper_ids_set: logging.debug(f"  Paper {current_paper_id} was already collected.")
        for ref_or_cit_list_key in ['references', 'citations']:
            for item in main_paper_api_response.get(ref_or_cit_list_key, []) or []: 
                item_id = item.get('paperId') 
                if item_id and item_id not in visited_for_processing: queue.append(item_id); visited_for_processing.add(item_id); logging.debug(f"    Added {ref_or_cit_list_key[:-1]} paper {item_id} to BFS queue.")
        if len(collected_papers_data) >= MAX_PAPERS_TO_COLLECT: logging.info(f"Reached maximum papers to collect ({MAX_PAPERS_TO_COLLECT})."); break
    return collected_papers_data

def save_to_csv(data_list, filename, headers):
    # (This function remains the same)
    if not data_list: logging.info("No data to save to CSV."); return
    logging.info(f"Saving {len(data_list)} papers to {filename}...")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for paper_data_dict in data_list:
                row_to_write = {}
                for key, value in paper_data_dict.items():
                    if key not in headers: continue
                    if isinstance(value, (list, dict)):
                        try: row_to_write[key] = json.dumps(value)
                        except TypeError as e: logging.error(f"JSON Error for paperId '{paper_data_dict.get('paperId')}', field '{key}': {e}. Value: {value}"); row_to_write[key] = "SERIALIZATION_ERROR"
                    else: row_to_write[key] = value
                writer.writerow(row_to_write)
        logging.info(f"Successfully saved data to {filename}")
    except IOError as e: logging.error(f"CSV Write Error for {filename}: {e}")
    except Exception as e: logging.error(f"Unexpected CSV writing error: {e}")

if __name__ == "__main__":
    logging.info("Starting NLP paper data collection script (v6 with isInfluential)...")
    print("-" * 70 + f"\nSemantic Scholar NLP Paper Crawler (v6 with isInfluential)\n" + "-" * 70)
    print(f"Configuration:\n  Min Citation: {MIN_CITATION_COUNT}, Max Papers: {MAX_PAPERS_TO_COLLECT}")
    print(f"  Year Range: {YEAR_RANGE_START}-{YEAR_RANGE_END if YEAR_RANGE_END else 'Present'}")
    print(f"  API Delay: {API_DELAY_SECONDS}s, Output: {OUTPUT_CSV_FILE}")
    print(f"  Paginated Endpoint Limit per call: {PAGINATED_ENDPOINT_LIMIT}")
    print(f"  Seed IDs: {len(SEED_PAPER_IDS)}, CSV Metadata Fields: {len(REQUESTED_METADATA_FIELDS)}")
    print(f"  CS FOS Category for acceptance: '{CS_FOS_CATEGORY}'")
    print(f"  Fetching 'isInfluential' flag for connections.")
    print("-" * 70)
    if API_KEY == "YOUR_SEMANTIC_SCHOLAR_API_KEY": print("WARNING: API_KEY is not set. Unauthenticated requests are SEVERELY rate-limited.\n" + "-" * 70)
    collected_papers_data_main = [] 
    try: collected_papers_data_main = fetch_papers_bfs()
    except Exception as e: logging.error(f"An error occurred during fetch_papers_bfs: {e}", exc_info=True)
    if collected_papers_data_main: save_to_csv(collected_papers_data_main, OUTPUT_CSV_FILE, REQUESTED_METADATA_FIELDS)
    logging.info("NLP paper data collection script (v6 - with isInfluential) finished.")
    final_count = len(collected_papers_data_main) if collected_papers_data_main is not None else 0
    print("-" * 70 + f"\nScript execution complete. Collected {final_count} papers.")
    print(f"Data saved to: {OUTPUT_CSV_FILE if (collected_papers_data_main and final_count > 0) else 'No data saved'}\n" + "-" * 70)

