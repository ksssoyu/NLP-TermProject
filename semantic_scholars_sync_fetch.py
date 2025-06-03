import requests
import time
import json
import csv
from collections import deque

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
    "3febb2bed8865945e7fddc99efd791887bb7e14f"   # ELMo
]

PAPER_FIELDS_PRIMARY_LIST = [
    'paperId',
    'title',
    'year',
    'authors',
    's2fieldsOfStudy',
    'citationCount',
    'citations',
    'references'
]

MAX_PAPERS_TO_FETCH = 20000
MIN_CITATION_COUNT = 170
PAPERS_PER_PAGE = 100
DELAY_SECONDS = 5

# New: Define the target Field of Study categories
TARGET_FOS_CATEGORIES = {
    "Computer Science",
    "Artificial Intelligence",
    "Information Retrieval",
    "Natural Language Processing"
}

def process_authors(authors_data):
    """Extracts author names and IDs from the Semantic Scholar API response."""
    authors_list = []
    if authors_data:
        for author in authors_data:
            authors_list.append({'name': author.get('author', {}).get('name'), 'authorId': author.get('authorId')})
    return authors_list

def process_citations_references(refs_data):
    """Extracts paper IDs and years from citations or references data."""
    refs_list = []
    if refs_data:
        for ref in refs_data:
            refs_list.append({'paperId': ref.get('paperId'), 'year': ref.get('year')})
    return refs_list

def fetch_paper_metadata(paper_id, fields):
    """Fetches metadata for a single paper from Semantic Scholar API."""
    base_url = "https://api.semanticscholar.org/graph/v1/paper/"
    params = {"fields": ",".join(fields)}
    try:
        response = requests.get(f"{base_url}{paper_id}", params=params)
        time.sleep(DELAY_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for paper ID {paper_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for paper ID {paper_id}: {e}")
        return None

def fetch_all_citing_papers(paper_id, fields, min_citations=0, limit=100):
    """Fetches all papers that cite a given paper from Semantic Scholar API using pagination."""
    base_url = "https://api.semanticscholar.org/graph/v1/paper/"
    all_citing_papers = []
    offset = 0

    while True:
        params = {"fields": ",".join(fields), "limit": limit, "offset": offset}
        try:
            response = requests.get(f"{base_url}{paper_id}/citations", params=params)
            time.sleep(DELAY_SECONDS)
            response.raise_for_status()
            data = response.json()
            citing_papers = [paper['citingPaper'] for paper in data.get('data', []) if paper['citingPaper'].get('citationCount', 0) >= min_citations]
            if not citing_papers:
                break
            all_citing_papers.extend(citing_papers)
            offset += limit
        except requests.exceptions.RequestException as e:
            print(f"Error fetching citing papers for paper ID {paper_id} at offset {offset}: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for citing papers of {paper_id} at offset {offset}: {e}")
            break
    return all_citing_papers

def collect_papers_bfs(seed_paper_ids, fields, min_citations, max_papers): #
    """Collects paper data using a breadth-first search approach."""
    collected_papers = [] #
    visited_paper_ids = set() #
    queue = deque([(paper_id, 0) for paper_id in seed_paper_ids]) # (paper_id, depth)

    while queue and len(collected_papers) < max_papers: #
        current_paper_id, depth = queue.popleft() #
        if current_paper_id in visited_paper_ids: #
            continue
        visited_paper_ids.add(current_paper_id) #

        paper_data = fetch_paper_metadata(current_paper_id, fields) #
        if paper_data:
            # Check citation count first
            if paper_data.get('citationCount', 0) >= min_citations: #
                # Extract s2fieldsOfStudy categories for validation
                paper_s2_categories = [f['category'] for f in paper_data.get('s2fieldsOfStudy', []) if 'category' in f] #

                # New: Validate if the paper is related to the target FOS categories
                is_relevant_by_fos = any(fos in TARGET_FOS_CATEGORIES for fos in paper_s2_categories)

                if is_relevant_by_fos:
                    # If relevant by FOS, then process and add the paper
                    processed_paper = {
                        'paperId': paper_data.get('paperId'), #
                        'title': paper_data.get('title'), #
                        'year': paper_data.get('year'), #
                        'authors': process_authors(paper_data.get('authors')), #
                        's2fieldsOfStudy': paper_s2_categories, # Use the categories extracted for validation
                        'citationCount': paper_data.get('citationCount'), #
                        'citations': process_citations_references(paper_data.get('citations')), #
                        'references': process_citations_references(paper_data.get('references')) #
                    }
                    collected_papers.append(processed_paper) #
                    print(f"Collected: {processed_paper.get('title')} (FoS relevant, Citations: {processed_paper.get('citationCount')})")


                    # Proceed to add citing papers to the queue if the current paper was added
                    if depth < 1: # Explore one level deeper for citing papers
                        citing_papers = fetch_all_citing_papers(current_paper_id, ['paperId', 'citationCount'], min_citations=min_citations, limit=PAPERS_PER_PAGE) #
                        for citing_paper in citing_papers:
                            if citing_paper['paperId'] not in visited_paper_ids and len(collected_papers) < max_papers: #
                                queue.append((citing_paper['paperId'], depth + 1)) #
                # else:
                    # Optional: print a message if a paper is skipped due to FOS
                    # print(f"Skipped (FoS): {paper_data.get('title')}")
            # else:
                # Optional: print a message if a paper is skipped due to citation count
                # print(f"Skipped (Citations): {paper_data.get('title')}, Citations: {paper_data.get('citationCount', 0)}")


    return collected_papers

if __name__ == "__main__":
    # Using MIN_CITATION_COUNT and MAX_PAPERS_TO_FETCH constants in the function call
    collected_data = collect_papers_bfs(INITIAL_PAPER_IDS, PAPER_FIELDS_PRIMARY_LIST, MIN_CITATION_COUNT, MAX_PAPERS_TO_FETCH) #

    if collected_data:
        csv_file_name = "nlp_paper_data.csv"
        with open(csv_file_name, 'w', newline='', encoding='utf-8') as csvfile: #
            # Ensure 'fieldsOfStudy' matches the key used in processed_paper ('s2fieldsOfStudy') or adjust as needed
            # The original script used 'fieldsOfStudy' in the CSV header for 's2fieldsOfStudy'.
            # Let's keep the CSV header as 'fieldsOfStudy' for consistency with the original output,
            # but ensure the data comes from processed_paper['s2fieldsOfStudy'].
            fieldnames = ['paperId', 'title', 'year', 'authors', 'fieldsOfStudy', 'citationCount', 'citations', 'references'] #
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames) #

            writer.writeheader() #
            for paper in collected_data:
                writer.writerow({ #
                    'paperId': paper['paperId'],
                    'title': paper['title'],
                    'year': paper['year'],
                    'authors': json.dumps(paper['authors'], ensure_ascii=False),
                    'fieldsOfStudy': json.dumps(paper['s2fieldsOfStudy'], ensure_ascii=False), # Data from 's2fieldsOfStudy'
                    'citationCount': paper['citationCount'],
                    'citations': json.dumps(paper['citations'], ensure_ascii=False),
                    'references': json.dumps(paper['references'], ensure_ascii=False)
                })
        print(f"\nSuccessfully saved {len(collected_data)} papers to {csv_file_name}") #
    else:
        print("No papers collected based on the criteria.") #