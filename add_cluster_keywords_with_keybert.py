"""
Add Cluster Keywords Using KeyBERT + TF-IDF Filtering

This script enhances per-year NLP topic cluster files by extracting representative keywords
from paper titles using KeyBERT, followed by TF-IDF-based filtering to retain unique and 
meaningful terms.

📌 Key Features:
- Uses KeyBERT (`all-MiniLM-L6-v2`) to extract candidate keywords from paper titles in each cluster
- Filters out domain-generic stopwords and English stopwords
- Applies TF-IDF filtering to keep distinctive terms across clusters
- Retains top-N (e.g., 5) final keywords per cluster
- Saves enriched cluster files with a new `"cluster_keywords"` field

📁 Input:
- Folder: `cluster_results_by_year/`
- JSON files with format:
  {
    "window": "2017_2019",
    "cluster_details": {
        "0": [{"title": "...", ...}, ...],
        ...
    }
  }

📁 Output:
- Folder: `cluster_with_keywords_by_year/`
- Same structure with additional:
  {
    "cluster_keywords": {
        "0": ["transformer", "qa", "dialogue", ...],
        ...
    }
  }

⚙️ Settings:
- CANDIDATE_KEYWORDS: Number of keywords extracted from KeyBERT (default: 15)
- FINAL_KEYWORDS: Number of final keywords saved after TF-IDF filtering (default: 5)
- DOMAIN_STOPWORDS: Custom NLP domain stopword set

Usage:
- Run with: `python add_cluster_keywords_with_keybert.py`
"""

import json
import os
from tqdm import tqdm
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 🔧 설정
INPUT_DIR = "cluster_results_by_year"     # 연도별 클러스터링 결과 폴더
OUTPUT_DIR = "cluster_with_keywords_by_year"   # 키워드 추가 결과 저장 폴더
CANDIDATE_KEYWORDS = 15  # 후보 키워드 수 증가
FINAL_KEYWORDS = 5

# ✅ 도메인 불용어 필터링
DOMAIN_STOPWORDS = {
    "neural", "language", "learning", "text", "model", "data", "approach",
    "paper", "method", "methods", "study", "results", "system"
}

# 📂 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 KeyBERT 모델 로드
kw_model = KeyBERT("all-MiniLM-L6-v2")

# 📂 입력 파일 순회
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith(".json"):
        continue

    input_path = os.path.join(INPUT_DIR, fname)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cluster_details = data.get("cluster_details", {})
    cluster_keywords_raw = {}
    corpus = []
    cluster_texts = {}

    for cluster_id, papers in tqdm(cluster_details.items(), desc=f"[KeyBERT] {fname}"):
        titles = [paper["title"] for paper in papers if paper.get("title")]
        text = " ".join(titles)
        cluster_texts[cluster_id] = text
        corpus.append(text)

        if not text.strip():
            cluster_keywords_raw[cluster_id] = []
            continue

        keywords = kw_model.extract_keywords(
            text,
            top_n=CANDIDATE_KEYWORDS + 10,
            stop_words='english',
            use_maxsum=True,
            nr_candidates=30,
            keyphrase_ngram_range=(1, 1)
        )
        filtered_keywords = [
            kw for kw, _ in keywords if kw.lower() not in DOMAIN_STOPWORDS
        ][:CANDIDATE_KEYWORDS]
        cluster_keywords_raw[cluster_id] = filtered_keywords

    # TF-IDF 분석 준비
    all_keywords = set(kw for kws in cluster_keywords_raw.values() for kw in kws)
    vectorizer = TfidfVectorizer(vocabulary=list(all_keywords))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    cluster_keywords_final = {}
    cluster_ids = list(cluster_texts.keys())

    for idx, cluster_id in enumerate(cluster_ids):
        row = tfidf_matrix[idx].toarray().flatten()
        keyword_scores = {feature_names[i]: row[i] for i in range(len(feature_names))}
        # 전체 평균보다 높은 TF-IDF를 갖는 단어만 남기기 (강한 필터링)
        mean_score = np.mean(tfidf_matrix.toarray(), axis=0)
        unusual_keywords = [
            kw for kw in cluster_keywords_raw[cluster_id]
            if kw in keyword_scores and keyword_scores[kw] > mean_score[feature_names.tolist().index(kw)]
        ]
        # 스코어 기준 상위 FINAL_KEYWORDS개 선택
        sorted_keywords = sorted(
            unusual_keywords,
            key=lambda kw: keyword_scores[kw],
            reverse=True
        )
        cluster_keywords_final[cluster_id] = sorted_keywords[:FINAL_KEYWORDS]

    # 🔹 원본에 키워드 추가
    data["cluster_keywords"] = cluster_keywords_final

    # 🔹 저장
    output_path = os.path.join(OUTPUT_DIR, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {output_path}")
