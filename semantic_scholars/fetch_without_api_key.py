import asyncio
import aiohttp
import csv
import time
from collections import deque
from tqdm.asyncio import tqdm_asyncio # For async-compatible progress bar
import random, requests


# --- Configuration ---
# Explicitly set API key to empty for public access
SEMANTIC_SCHOLAR_API_KEY = ""

# --- Collection Strategy Parameters ---
# (Keep these as you had them or adjust as needed)
INITIAL_PAPER_IDS = [
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

MIN_CITATION_THRESHOLD = 170
MAX_PAPERS_TO_COLLECT = 10000 
MAX_DEPTH = 2

# --- API & CSV Configuration ---
# PAPER_FIELDS_PRIMARY = [
#     'paperId', 'title', 'year', 'authors.name', 'authors.authorId',
#     's2FieldsOfStudy', 'citationCount',
#     'citations.paperId', 'citations.citationCount',
#     'references.paperId', 'references.citationCount'
# ]
PAPER_FIELDS_PRIMARY = [
    'paperId',
    'title',
    'year',
    'citationCount' # Added this basic important field
]
CSV_FILENAME = "nlp_influential_papers_no_api_key.csv" # Changed filename slightly

# --- Concurrency and Rate Limiting Control for NO API KEY ---
# Public limit: 100 requests per 5 minutes (300 seconds) = 1 request every 3 seconds.
# --- Configuration ---
SEMANTIC_SCHOLAR_API_KEY = ""
# ... (INITIAL_PAPER_IDS, MIN_CITATION_THRESHOLD, etc.) ...

# --- Exponential Backoff Parameters ---
MAX_FETCH_RETRIES = 6  # Max retries for a single paper fetch on 429 or 5xx (e.g., 3-5)
INITIAL_BACKOFF_DELAY_SECONDS  = 4  # Initial delay for the first retry (e.g., 5-10s for public API)
MAX_BACKOFF_DELAY_SECONDS = 60    # Max delay for a single retry sleep (e.g., 60-120s)

# --- Concurrency and Rate Limiting Control for NO API KEY ---
MAX_CONCURRENT_REQUESTS = 1
REQUEST_BASE_DELAY = 5.0 # Base delay after a successful fetch or non-retried error

# --- Helper Functions (Async & Processing) ---
async def fetch_paper_details_async(session, paper_id, api_key_to_use, semaphore):
    # This async with semaphore ensures that only MAX_CONCURRENT_REQUESTS
    # are actively running the code within this block.
    async with semaphore:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/"
        fields_param = ",".join(PAPER_FIELDS_PRIMARY)
        url = f"{base_url}{paper_id}?fields={fields_param}"
        headers = {}
        if api_key_to_use:  # This will be false if SEMANTIC_SCHOLAR_API_KEY is ""
            headers['x-api-key'] = api_key_to_use

        for attempt in range(MAX_FETCH_RETRIES + 1): # 0 is the first try, 1 to MAX_FETCH_RETRIES are retries
            try:
                # For NO API KEY mode with MAX_CONCURRENT_REQUESTS = 1:
                # If this is the first attempt for *this paper*, the REQUEST_BASE_DELAY
                # should have been applied *after the previous paper's fetch completed*.
                # If this is a retry (attempt > 0), we will have already slept due to backoff.
                
                async with session.get(url, headers=headers, timeout=30) as response: # Added a 30s timeout
                    # Manually check for 429 to cleanly trigger our retry logic
                    if response.status == 429:
                        await response.text()  # Consume the response body to prevent warnings
                        # Raise an error to be caught by our ClientResponseError handler
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message="Rate limit hit (triggering retry)",
                            headers=response.headers,
                        )
                    
                    response.raise_for_status() # Raise HTTPError for other bad responses (4xx or 5xx)
                    data = await response.json()
                    if attempt == 0 and REQUEST_BASE_DELAY > 0:
                        await asyncio.sleep(REQUEST_BASE_DELAY)
                    # If successful on the first actual try for this paper (not a retry attempt),
                    # apply the REQUEST_BASE_DELAY to space out from the next *new* paper's request.
                    
                    return data

            except (aiohttp.ClientResponseError, asyncio.TimeoutError, aiohttp.ClientError) as e:
                error_status_code = e.status if hasattr(e, 'status') else None # For ClientResponseError
                error_type_for_log = "Rate Limit (429)" if error_status_code == 429 else \
                                  f"Server Error ({error_status_code})" if error_status_code and 500 <= error_status_code <= 599 else \
                                  "Timeout Error" if isinstance(e, asyncio.TimeoutError) else \
                                  f"Client/Connection Error ({error_status_code or type(e).__name__})"

                # Retry only for 429 (Rate Limit), 5xx (Server Errors), Timeouts, or specific ClientErrors like connection errors
                if error_status_code == 429 or \
                   (error_status_code and 500 <= error_status_code <= 599) or \
                   isinstance(e, asyncio.TimeoutError) or \
                   isinstance(e, aiohttp.ClientConnectionError): # Added ClientConnectionError

                    if attempt < MAX_FETCH_RETRIES:
                        # Calculate exponential backoff delay with jitter
                        backoff_duration = min(MAX_BACKOFF_DELAY_SECONDS, INITIAL_BACKOFF_DELAY_SECONDS * (2 ** attempt))
                        jitter = random.uniform(0, min(2.0, backoff_duration * 0.1)) # Jitter up to 2s or 10%
                        actual_sleep = backoff_duration + jitter
                        
                        print(f"{error_type_for_log} for paper {paper_id}. Attempt {attempt + 1}/{MAX_FETCH_RETRIES + 1}. Retrying in {actual_sleep:.2f}s...")
                        await asyncio.sleep(actual_sleep)
                        continue # Go to the next attempt in the for loop
                    else:
                        print(f"{error_type_for_log} for paper {paper_id}. Max retries ({MAX_FETCH_RETRIES}) reached. Giving up on this paper.")
                        # Apply REQUEST_BASE_DELAY even after giving up to maintain overall pace
                        if REQUEST_BASE_DELAY > 0: await asyncio.sleep(REQUEST_BASE_DELAY)
                        return None
                else: # Non-retriable client errors (e.g., 400, 401, 403, 404)
                    error_message = e.message if hasattr(e, 'message') and e.message else str(e)
                    print(f"HTTP error {error_status_code or type(e).__name__} ({error_message}) for paper {paper_id}. Not retrying.")
                    if REQUEST_BASE_DELAY > 0: await asyncio.sleep(REQUEST_BASE_DELAY) # Maintain pace
                    return None
            except Exception as e: # Catch-all for other unexpected errors during the fetch attempt
                if attempt < MAX_FETCH_RETRIES:
                    backoff_duration = min(MAX_BACKOFF_DELAY_SECONDS, INITIAL_BACKOFF_DELAY_SECONDS * (2 ** attempt))
                    jitter = random.uniform(0, min(2.0, backoff_duration * 0.1))
                    actual_sleep = backoff_duration + jitter
                    print(f"Unexpected generic error fetching {paper_id}: {str(e)}. Attempt {attempt + 1}/{MAX_FETCH_RETRIES + 1}. Retrying in {actual_sleep:.2f}s...")
                    await asyncio.sleep(actual_sleep)
                    continue
                else:
                    print(f"Unexpected generic error fetching {paper_id} after max retries: {str(e)}. Giving up.")
                    if REQUEST_BASE_DELAY > 0: await asyncio.sleep(REQUEST_BASE_DELAY) # Maintain pace
                    return None
        
        # If the loop completes, all retries for this paper_id have failed
        print(f"Failed to fetch {paper_id} after all {MAX_FETCH_RETRIES + 1} attempts (exhausted retry loop).")
        # Apply REQUEST_BASE_DELAY even after giving up to maintain overall pace before next paper
        if REQUEST_BASE_DELAY > 0: await asyncio.sleep(REQUEST_BASE_DELAY)
        return None

def process_paper_data_for_csv(paper_data_raw):
    if not paper_data_raw: return None
    authors_info = paper_data_raw.get('authors', [])
    authors_names = "; ".join([author.get('name', 'N/A') for author in authors_info])
    concepts_list = []
    for field_group in paper_data_raw.get('s2FieldsOfStudy', []):
        if field_group and field_group.get('category'):
            concepts_list.append(field_group['category'])
    concepts = "; ".join(sorted(list(set(concepts_list))))

    citing_paper_ids = "; ".join(
        sorted(list(set(c['paperId'] for c in paper_data_raw.get('citations', []) if c.get('paperId'))))
    )
    reference_paper_ids = "; ".join(
        sorted(list(set(r['paperId'] for r in paper_data_raw.get('references', []) if r.get('paperId'))))
    )

    return {
        'paper_id': paper_data_raw.get('paperId'),
        'title': paper_data_raw.get('title'),
        'year': paper_data_raw.get('year'),
        'authors': authors_names,
        'concepts': concepts,
        'citation_count_self': paper_data_raw.get('citationCount', 0),
        'citing_paper_ids': citing_paper_ids,
        'reference_paper_ids': reference_paper_ids
    }

# --- Main Async Script ---
async def async_main_iterative():
    print("--- Running in NO API KEY mode ---")
    print(f"Targeting ~1 request every {REQUEST_BASE_DELAY} seconds due to public rate limits.")
    print("This will be slow. Estimated time for 10,000 papers: ~8-9 hours.")
    print("Make sure your computer does not go to sleep during this process.")
    print("------------------------------------")
    await asyncio.sleep(5) # Give user time to read the warning

    collected_papers_for_csv = []
    processed_paper_ids = set()
    queue = deque([(paper_id, 0) for paper_id in INITIAL_PAPER_IDS])
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) # Set to 1

    # Session headers will be empty as SEMANTIC_SCHOLAR_API_KEY is ""
    async with aiohttp.ClientSession() as session:
        with tqdm_asyncio(total=MAX_PAPERS_TO_COLLECT, desc="Collecting Papers (Slow Mode)") as pbar:
            while queue and len(collected_papers_for_csv) < MAX_PAPERS_TO_COLLECT:
                paper_id_to_fetch, current_depth = queue.popleft()

                if paper_id_to_fetch in processed_paper_ids:
                    continue

                if current_depth > MAX_DEPTH:
                    pbar.set_description(f"Max depth {MAX_DEPTH} for {paper_id_to_fetch[:10]}")
                    continue

                pbar.set_description(f"Fetching L{current_depth}: {paper_id_to_fetch[:10]}")
                # Pass SEMANTIC_SCHOLAR_API_KEY (which is "") to the fetch function
                paper_details_raw = await fetch_paper_details_async(session, paper_id_to_fetch, SEMANTIC_SCHOLAR_API_KEY, semaphore)

                if not paper_details_raw:
                    processed_paper_ids.add(paper_id_to_fetch)
                    continue

                csv_row_data = process_paper_data_for_csv(paper_details_raw)
                if csv_row_data:
                    collected_papers_for_csv.append(csv_row_data)
                    processed_paper_ids.add(paper_id_to_fetch)
                    pbar.update(1)
                    pbar.set_postfix_str(f"Collected: {len(collected_papers_for_csv)}")

                if current_depth < MAX_DEPTH and len(collected_papers_for_csv) < MAX_PAPERS_TO_COLLECT:
                    for citing_paper_info in paper_details_raw.get('citations', []):
                        if len(collected_papers_for_csv) >= MAX_PAPERS_TO_COLLECT: break
                        if citing_paper_info.get('paperId') and citing_paper_info.get('paperId') not in processed_paper_ids:
                            raw_citing_citation_count = citing_paper_info.get('citationCount')
                            
                            if raw_citing_citation_count is None or (raw_citing_citation_count is not None and raw_citing_citation_count >= MIN_CITATION_THRESHOLD):
                                queue.append((citing_paper_info['paperId'], current_depth + 1))
                    
                    for ref_paper_info in paper_details_raw.get('references', []):
                        if len(collected_papers_for_csv) >= MAX_PAPERS_TO_COLLECT: break
                        if ref_paper_info.get('paperId') and ref_paper_info.get('paperId') not in processed_paper_ids:
                            raw_ref_citation_count = ref_paper_info.get('citationCount')

                            if raw_ref_citation_count is None or (raw_ref_citation_count is not None and raw_ref_citation_count >= MIN_CITATION_THRESHOLD):
                                queue.append((ref_paper_info['paperId'], current_depth + 1))

    if not collected_papers_for_csv:
        print("No data collected. Exiting.")
        return

    csv_header = [
        'paper_id', 'title', 'year', 'authors', 'concepts',
        'citation_count_self',
        'citing_paper_ids', 'reference_paper_ids'
    ]
    try:
        with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(collected_papers_for_csv)
        print(f"\nSuccessfully saved {len(collected_papers_for_csv)} papers to {CSV_FILENAME}")
    except IOError:
        print(f"Error writing to CSV file {CSV_FILENAME}")

if __name__ == "__main__":
    print(f"Script starting. Max papers: {MAX_PAPERS_TO_COLLECT}. Max depth: {MAX_DEPTH}. Min citation threshold: {MIN_CITATION_THRESHOLD}.")
    
    start_time = time.time()
    asyncio.run(async_main_iterative())
    end_time = time.time()
    print(f"Script finished in {end_time - start_time:.2f} seconds.")