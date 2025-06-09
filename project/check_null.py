import pandas as pd

# CSV 파일 로드
df = pd.read_csv("merged_nlp_papers_for_enrichment.csv")

# 세 컬럼이 모두 notna()인 행만 필터링
filtered_df = df[df["references_data"].notna() & df["citations_data"].notna() & df["embedding"].notna()]

# 개수 출력
print(f"✅ references, citations, embedding 모두 존재하는 논문 수: {len(filtered_df)}")
