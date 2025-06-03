import asyncio
import aiohttp
import csv
import time
from collections import deque
from tqdm.asyncio import tqdm_asyncio

# --- Configuration ---
SEMANTIC_SCHOLAR_API_KEY = ""  # !!! YOUR API KEY HERE !!!

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

# --- Collection Strategy Parameters ---
# Minimum citation count for a related paper to be considered "significant"
# and added to the queue for further processing and to the dataset.
MIN_CITATION_THRESHOLD = 170 # Adjust this threshold as needed

# Maximum number of papers to collect in total (including seeds)
MAX_PAPERS_TO_COLLECT = 10000 # Adjust as needed

# Maximum depth of traversal from seed papers
# Depth 0 = seed papers only
# Depth 1 = seed papers + significant papers they cite or are cited by
# Depth 2 = Depth 1 + significant papers connected to Depth 1 papers
MAX_DEPTH = 2 # Adjust as needed

# --- API & CSV Configuration ---
PAPER_FIELDS_PRIMARY = [ # Fields for papers we decide to fully process and add to our dataset
    'paperId', 'title', 'year', 'authors.name', 'authors.authorId',
    's2FieldsOfStudy.category', 'citationCount', 'abstract',
    'citations.paperId', 'citations.title', 'citations.year', 'citations.citationCount', # Info about citing papers
    'references.paperId', 'references.title', 'references.year', 'references.citationCount' # Info about referenced papers
]
# When fetching related papers (citations/references of a primary paper),
# we need their citationCount to decide if they are significant.
# The fields above in 'citations.*' and 'references.*' already cover this.

CSV_FILENAME = "nlp_influential_papers.csv"
MAX_CONCURRENT_REQUESTS = 5 # Adjust based on API key (e.g., 1-2 no key, 10-100 with key)
REQUEST_BASE_DELAY = 0.1 # s (can be 0 with a good API key and high concurrency)

# --- Helper Functions (Async & Processing) ---
async def fetch_paper_details_async(session, paper_id, api_key, semaphore, pbar_item_desc="Fetching"):
    async with semaphore:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/"
        fields_param = ",".join(PAPER_FIELDS_PRIMARY)
        url = f"{base_url}{paper_id}?fields={fields_param}"
        headers = {}
        if api_key: headers['x-api-key'] = api_key

        try:
            # print(f"Queueing fetch for: {paper_id[:10]}...") # Debug
            if pbar_item_desc:
                # This part is tricky with concurrent tqdm_asyncio.gather,
                # as individual task descriptions might not update well on a single bar.
                # The main progress bar from gather is usually sufficient.
                pass
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                if MAX_CONCURRENT_REQUESTS <= 2 and REQUEST_BASE_DELAY > 0:
                    await asyncio.sleep(REQUEST_BASE_DELAY)
                return data
        except aiohttp.ClientResponseError as e:
            print(f"HTTP error for paper {paper_id}: {e.status} - {e.message}")
            if e.status == 429:
                print("Rate limit hit hard. Waiting 60s...")
                await asyncio.sleep(60)
        except Exception as e:
            print(f"Generic error fetching {paper_id}: {e}")
        return None

def process_paper_data_for_csv(paper_data_raw):
    if not paper_data_raw: return None
    authors_info = paper_data_raw.get('authors', [])
    authors_names = "; ".join([author.get('name', 'N/A') for author in authors_info])
    # concepts = "; ".join(sorted(list(set(field['category'] for field in paper_data_raw.get('s2FieldsOfStudy', []) if field.get('category'))))) # Unique sorted
    concepts_list = []
    for field_group in paper_data_raw.get('s2FieldsOfStudy', []):
        if field_group and field_group.get('category'): # s2FieldsOfStudy is a list of dicts
            concepts_list.append(field_group['category'])
    concepts = "; ".join(sorted(list(set(concepts_list))))


    # Store only IDs of significant citations/references, or all if not filtering here
    # For this version, we are already filtering which *papers* get added to the dataset.
    # The 'citations' and 'references' in the CSV will be for the *primary* paper in that row.
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
        'abstract': paper_data_raw.get('abstract', ''),
        'citing_paper_ids': citing_paper_ids, # All citing papers' IDs for this specific paper
        'reference_paper_ids': reference_paper_ids # All referenced papers' IDs for this specific paper
    }

# --- Main Async Script ---
async def async_main_iterative():
    collected_papers_for_csv = [] # List to store dicts for CSV rows
    processed_paper_ids = set()   # IDs of papers whose details have been fetched and added to collected_papers_for_csv
    
    # Queue stores (paper_id, current_depth)
    # Using collections.deque as a simple queue
    queue = deque([(paper_id, 0) for paper_id in INITIAL_PAPER_IDS])
    
    # Add initial seed IDs to processed_paper_ids to ensure they are fetched if they are in queue
    # but they will be primary targets.
    # Effectively, seed papers bypass initial significance checks if they are roots.
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY

    async with aiohttp.ClientSession(headers=headers) as session:
        # Progress bar for the number of papers we aim to collect
        with tqdm_asyncio(total=MAX_PAPERS_TO_COLLECT, desc="Collecting Papers") as pbar:
            
            while queue and len(collected_papers_for_csv) < MAX_PAPERS_TO_COLLECT:
                paper_id_to_fetch, current_depth = queue.popleft()

                if paper_id_to_fetch in processed_paper_ids:
                    continue # Already processed this paper fully

                if current_depth > MAX_DEPTH:
                    pbar.set_description(f"Max depth {MAX_DEPTH} reached for branch of {paper_id_to_fetch[:10]}")
                    continue

                pbar.set_description(f"Fetching L{current_depth}: {paper_id_to_fetch[:10]}")
                paper_details_raw = await fetch_paper_details_async(session, paper_id_to_fetch, None, semaphore) # API key in session headers

                if not paper_details_raw:
                    processed_paper_ids.add(paper_id_to_fetch) # Mark as processed even if fetch failed to avoid retries
                    # pbar.update(1) # Only update pbar when a paper is successfully added to dataset
                    continue

                # Add the current paper to our dataset
                # Seed papers are always added. Others are added if they were deemed significant by the previous step.
                csv_row_data = process_paper_data_for_csv(paper_details_raw)
                if csv_row_data:
                    collected_papers_for_csv.append(csv_row_data)
                    processed_paper_ids.add(paper_id_to_fetch)
                    pbar.update(1) # Increment progress bar for each paper added to the dataset
                    pbar.set_postfix_str(f"Collected: {len(collected_papers_for_csv)}")


                # If we haven't reached max depth, explore citations and references
                if current_depth < MAX_DEPTH and len(collected_papers_for_csv) < MAX_PAPERS_TO_COLLECT :
                    
                    # Add significant CITING papers to the queue
                    for citing_paper_info in paper_details_raw.get('citations', []):
                        if len(collected_papers_for_csv) >= MAX_PAPERS_TO_COLLECT: break
                        if citing_paper_info.get('paperId') and citing_paper_info.get('paperId') not in processed_paper_ids:
                            # The 'citationCount' here is for the citing_paper_info itself
                            if citing_paper_info.get('citationCount', 0) >= MIN_CITATION_THRESHOLD:
                                queue.append((citing_paper_info['paperId'], current_depth + 1))
                    
                    # Add significant REFERENCED papers to the queue
                    for ref_paper_info in paper_details_raw.get('references', []):
                        if len(collected_papers_for_csv) >= MAX_PAPERS_TO_COLLECT: break
                        if ref_paper_info.get('paperId') and ref_paper_info.get('paperId') not in processed_paper_ids:
                            # The 'citationCount' here is for the ref_paper_info itself
                            if ref_paper_info.get('citationCount', 0) >= MIN_CITATION_THRESHOLD:
                                queue.append((ref_paper_info['paperId'], current_depth + 1))
                
                # Small breather to avoid overwhelming the queue processing loop itself if fetches are super fast
                await asyncio.sleep(0.01)


    if not collected_papers_for_csv:
        print("No data collected. Exiting.")
        return

    csv_header = [
        'paper_id', 'title', 'year', 'authors', 'concepts',
        'citation_count_self', 'abstract',
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
    print(f"Script starting. Aiming to collect up to {MAX_PAPERS_TO_COLLECT} papers.")
    print(f"Seed papers: {len(INITIAL_PAPER_IDS)}. Max depth: {MAX_DEPTH}. Min citation threshold for expansion: {MIN_CITATION_THRESHOLD}.")
    
    start_time = time.time()
    asyncio.run(async_main_iterative())
    end_time = time.time()
    print(f"Script finished in {end_time - start_time:.2f} seconds.")