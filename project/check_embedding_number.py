import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("merged_nlp_papers_for_enrichment.csv")

# 1. 전체 논문 수 (paperId 개수)
total_papers = df["paperId"].nunique()

# 2. embedding이 비어 있지 않은 논문 수
embedding_filled = df["embedding"].notna().sum()

print(f"📄 총 논문 수 (paperId): {total_papers}")
print(f"🧠 임베딩 존재 논문 수: {embedding_filled}")
print(f"⚖️ 임베딩 비율: {embedding_filled / total_papers:.2%}")
