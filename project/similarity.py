import pandas as pd
import itertools
import math
import json
import csv

# 🔧 JSON 파싱 함수들
def extract_paper_ids(refs_json_str):
    try:
        if not refs_json_str or refs_json_str.strip() == "":
            return []
        refs_list = json.loads(refs_json_str)
        return [ref['paperId'] for ref in refs_list if 'paperId' in ref]
    except Exception as e:
        print("⚠️ references_data 파싱 에러:", e)
        return []

def extract_citer_ids(citations_json_str):
    try:
        if not citations_json_str or citations_json_str.strip() == "":
            return []
        citer_list = json.loads(citations_json_str)
        return [c['paperId'] for c in citer_list if 'paperId' in c]
    except Exception as e:
        print("⚠️ citations_data 파싱 에러:", e)
        return []

# ✅ Step 1: CSV 로드
df = pd.read_csv("merged_nlp_papers_for_enrichment.csv")

# ✅ Step 2: references_data, citations_data 파싱
# ✅ NaN 방지를 위해 문자열로 변환
df['citations_data'] = df['citations_data'].fillna("").astype(str)
df['references_data'] = df['references_data'].fillna("").astype(str)

# ✅ 이후 파싱 적용
df['references_data'] = df['references_data'].apply(extract_paper_ids)
df['citations_data'] = df['citations_data'].apply(extract_citer_ids)

# ✅ Step 3: 논문 ID ➝ 참조/피인용 집합 생성
paper_refs = {row['paperId']: set(row['references_data']) for _, row in df.iterrows()}
paper_citers = {row['paperId']: set(row['citations_data']) for _, row in df.iterrows()}

# ✅ Step 4: 유사도 계산 및 CSV 저장
output_file = "nlp_papers_similarity.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['paper_a', 'paper_b', 'bc_score', 'cc_score', 'combined_score', 'common_refs', 'common_citers']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    paper_ids = list(paper_refs.keys())

    for paper_a, paper_b in itertools.combinations(paper_ids, 2):
        refs_a, refs_b = paper_refs[paper_a], paper_refs[paper_b]
        citers_a, citers_b = paper_citers.get(paper_a, set()), paper_citers.get(paper_b, set())

        # Bibliographic Coupling
        bc_count = len(refs_a & refs_b)
        bc_score = bc_count / math.sqrt(len(refs_a) * len(refs_b)) if refs_a and refs_b else 0

        # Co-citation
        cc_count = len(citers_a & citers_b)
        cc_score = cc_count / math.sqrt(len(citers_a) * len(citers_b)) if citers_a and citers_b else 0

        # 결합 점수 (alpha 조정 가능)
        alpha = 0.1
        combined_score = alpha * cc_score + (1 - alpha) * bc_score

        writer.writerow({
            'paper_a': paper_a,
            'paper_b': paper_b,
            'bc_score': bc_score,
            'cc_score': cc_score,
            'combined_score': combined_score,
            'common_refs': bc_count,
            'common_citers': cc_count
        })

print(f"✅ 유사도 계산 완료! 결과는 '{output_file}'에 저장됨.")
