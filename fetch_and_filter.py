import pandas as pd
import numpy as np
import re
import aiohttp
import asyncio
from tqdm import tqdm
import nest_asyncio
import ast # For safely evaluating string representations of lists
import random

nest_asyncio.apply()

BASE_URL = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "ppco915@gmail.com"} # IMPORTANT: Replace with your actual email!

MAX_RESULTS_PER_LEVEL = 200
MAX_TOTAL_PAPERS = 150000
MIN_CITATION_THRESHOLD = 6

# 논문 정보를 저장할 dict: key = id, value = 논문 정보
paper_data = {}
sem = asyncio.Semaphore(8)

async def fetch_json(session, url, retries=5, delay=2):
    for attempt in range(retries):
        try:
            async with sem:
                async with session.get(url, headers=HEADERS) as response:
                    if response.status == 429:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    response.raise_for_status() # Raise an exception for bad status codes
                    return await response.json()
        except aiohttp.ClientError as e:
            print(f"Client error fetching {url}: {e}")
        except Exception as e:
            print(f"Unexpected error fetching {url}: {e}")
        await asyncio.sleep(delay * (attempt + 1)) # Exponential backoff
    print(f"Failed to fetch {url} after {retries} attempts.")
    return {}

def extract_paper_info(paper_json, existing_author_ids):
    paper_id = paper_json["id"].split("/")[-1]
    title = paper_json.get("title", "")
    citation_count = paper_json.get("cited_by_count", 0)
    publication_year = paper_json.get("publication_year")
    if publication_year is None or publication_year < 2015:
        return None 
    
    authors = []
    author_ids = []
    institutions = []
    for auth in paper_json.get("authorships", []):
        author = auth.get("author", {})
        name = author.get("display_name")
        aid = author.get("id")
        if name and aid:
            aid_stripped = aid.split("/")[-1]
            if aid_stripped not in existing_author_ids: # Only add if not already seen in the set
                authors.append(name)
                author_ids.append(aid_stripped)
                existing_author_ids.add(aid_stripped) # Add to the set
            else: # If author already exists, just append their name/ID without re-adding to set
                authors.append(name)
                author_ids.append(aid_stripped)


        institutions_info = auth.get("institutions", [])
        for institution in institutions_info:
            institution_name = institution.get("display_name", "Unknown")
            institutions.append(institution_name)

    concepts = [c["display_name"] for c in paper_json.get("concepts", [])]
    keywords = [k["display_name"] for k in paper_json.get("keywords", [])]

    # --- NEW: Extract referenced works IDs ---
    # OpenAlex provides 'referenced_works' as a list of OpenAlex URLs
    referenced_works_urls = paper_json.get("referenced_works", [])
    # Extract just the IDs from the URLs (e.g., 'W1234567890')
    referenced_work_ids = [url.split('/')[-1] for url in referenced_works_urls]

    return {
        "id": paper_id,
        "title": title,
        "authors": "; ".join(authors),
        "author_ids": "; ".join(author_ids),
        "institutions": "; ".join(institutions),
        "citations": citation_count,
        "concepts": "; ".join(concepts) + "; ".join(keywords),
        "referenced_works": "; ".join(referenced_work_ids),
        "publication_year": publication_year
    }

async def get_citing_works(session, work_id, existing_author_ids, max_results=MAX_RESULTS_PER_LEVEL):
    citing = []
    cursor = "*"
    while len(citing) < max_results:
        url = f"{BASE_URL}?filter=cites:{work_id}&per-page=50&cursor={cursor}"
        data = await fetch_json(session, url)
        if not data or 'results' not in data:
            break
        for result in data["results"]:
            pid = result["id"].split("/")[-1]
            if pid not in paper_data: # Only add new papers
                info = extract_paper_info(result, existing_author_ids)
                if info:
                    paper_data[pid] = info
                citing.append(pid)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return citing

async def fetch_and_expand(session, work_id, existing_author_ids):
    # 1. 이 논문을 인용한 논문들을 가져옵니다 (현재 방식과 동일)
    citing_papers = await get_citing_works(session, work_id, existing_author_ids)

    # 2. 이 논문 자체의 상세 정보를 가져와서 참고문헌 목록을 확인합니다.
    # (이미 paper_data에 없거나 referenced_works 정보가 없을 경우)
    if work_id not in paper_data or not paper_data[work_id].get('referenced_works'):
        detailed_work_url = f"{BASE_URL}/{work_id}"
        detailed_work_json = await fetch_json(session, detailed_work_url)
        if detailed_work_json:
            info = extract_paper_info(detailed_work_json, existing_author_ids)
            if info:
                paper_data[work_id] = info
            

    # 3. 만약 이 논문의 참고문헌 목록이 있다면, 해당 참고문헌들도 수집합니다.
    if work_id in paper_data and paper_data[work_id].get('referenced_works'):
        referenced_ids_str = paper_data[work_id]['referenced_works']
        # 세미콜론으로 구분된 문자열을 리스트로 변환
        referenced_ids = [ref.strip() for ref in referenced_ids_str.split(';') if ref.strip()]
        for ref_id in referenced_ids:
            if len(paper_data) >= MAX_TOTAL_PAPERS: # 총 논문 수 한계에 도달하면 중단
                break
            if ref_id not in paper_data: # 아직 수집되지 않은 논문만 추가 수집
                ref_url = f"{BASE_URL}/{ref_id}"
                ref_json = await fetch_json(session, ref_url)
                if ref_json:
                    info = extract_paper_info(ref_json, existing_author_ids)
                    if info : 
                        paper_data[ref_id] = info

    # 4. (기존 로직 유지) 인용한 논문들의 인용 논문들도 확장합니다.
    for citer_id in citing_papers:
        if len(paper_data) >= MAX_TOTAL_PAPERS: # 총 논문 수 한계에 도달하면 중단
            break
        # 깊이 확장을 제한하여 과도한 수집 방지
        await get_citing_works(session, citer_id, existing_author_ids, max_results=10)


async def crawl_nlp_papers():
    # "T10181" : nlp techniques
    # "C204321447" : nlp
    # Using a broad NLP concept ID for initial crawl
    concept_id = "C204321447" # Natural Language Processing
    # Or, if you have a specific initial paper ID you want to start from, use that
    # initial_paper_id = "W12345" # Example: replace with a real OpenAlex Work ID if starting from a specific paper

    url = f"{BASE_URL}?filter=concepts.id:{concept_id}&per-page=50&cursor=*"
    # If starting from an initial paper, fetch it first and then expand
    # url = f"{BASE_URL}/{initial_paper_id}"

    existing_author_ids = set()

    async with aiohttp.ClientSession() as session:
        # Fetch initial set of papers (either by concept or a specific paper)
        data = await fetch_json(session, url)
        if not data or 'results' not in data:
            print("Failed to fetch initial papers.")
            return

        base_works_list = data["results"] # For concept search
        # If you were starting from a single paper: base_works_list = [data]

        tasks = []
        for work in base_works_list:
            work_id = work["id"].split("/")[-1]
            if work_id not in paper_data:
                info = extract_paper_info(work, existing_author_ids)
                if info:
                    paper_data[work_id] = info
                    tasks.append(fetch_and_expand(session, work_id, existing_author_ids))


        # Execute async tasks
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Crawling papers"):
            await f
            if len(paper_data) >= MAX_TOTAL_PAPERS:
                print(f"Reached MAX_TOTAL_PAPERS ({MAX_TOTAL_PAPERS}). Stopping crawl.")
                break
 
# Run the crawler
if __name__ == "__main__":
    asyncio.run(crawl_nlp_papers())

# --- Data Processing and Saving ---
# paper_data → 리스트로 변환
df = pd.DataFrame(paper_data.values())

# NLP 필터링 (기존 코드 유지)
nlp_keywords = [
    'natural language processing', 'nlp', 'text mining', 'sentiment analysis',
    'named entity recognition', 'ner', 'topic modeling', 'word embedding',
    'language model', 'machine translation', 'chatbot', 'dialogue system',
    'speech recognition', 'text classification', 'information extraction',
    'summarization', 'linguistics', 'computational linguistics',
    'natural language understanding', 'nlu', 'natural language generation', 'nlg',
    'corpus', 'syntax', 'semantics', 'pragmatics', 'discourse analysis',
    'part-of-speech tagging', 'pos tagging', 'parsing', 'grammar induction',
    'neural machine translation', 'seq2seq', 'attention mechanism', 'transformer',
    'bert', 'gpt', 'roberta', 't5', 'elmo', 'xlnet', 'tokenization', 'lemma',
    'stemming', 'stop words', 'tfidf', 'bag-of-words', 'skip-gram', 'cbow',
    'recurrent neural network', 'rnn', 'long short-term memory', 'lstm',
    'gated recurrent unit', 'gru', 'convolutional neural network', 'cnn',
    'sequence tagging', 'relation extraction', 'question answering',
    'summarization', 'text generation', 'stance detection', 'emotion recognition',
    'dialogue management', 'voice assistant', 'knowledge graph', 'ontologies',
    'wordnet', 'frame semantics', 'dependency parsing', 'constituency parsing',
    'coreference resolution', 'anaphora resolution', 'multi-modal nlp',
    'cross-lingual nlp', 'low-resource nlp', 'active learning for nlp',
    'reinforcement learning for nlp', 'transfer learning for nlp'
]

df['concepts_str'] = df['concepts'].astype(str).fillna('')
pattern = r'\b(' + '|'.join(re.escape(k) for k in nlp_keywords) + r')\b'
df_nlp_papers = df[df['concepts_str'].str.contains(pattern, case=False, na=False)].copy() # Use .copy() to avoid SettingWithCopyWarning
df_nlp_papers = df_nlp_papers.drop(columns=['concepts_str'])

print(f"Original DataFrame size (after crawl): {len(df)} rows")
print(f"Filtered DataFrame size (NLP papers): {len(df_nlp_papers)} rows")

# Save the NLP papers to CSV
output_columns = ["id", "title", "authors", "author_ids", "institutions",
                  "citations", "concepts", "referenced_works", "publication_year"]

# Ensure all output columns exist in df_nlp_papers, fill missing with empty string/NaN
for col in output_columns:
    if col not in df_nlp_papers.columns:
        df_nlp_papers[col] = '' # Or np.nan

# df_nlp_papers = df_nlp_papers[df_nlp_papers['ci tations'].astype(int) >= MIN_CITATION_THRESHOLD]
df_nlp_papers.to_csv("nlp_papers.csv", index=False, encoding="utf-8")

print(f"✅ 총 수집 논문 수: {len(paper_data)}개.")
print(f"✅ 필터링된 NLP 논문 수: {len(df_nlp_papers)}개.")
print(f"✅ 필터링된 NLP 논문 정보가 'nlp_papers.csv' 파일로 저장 완료.")

