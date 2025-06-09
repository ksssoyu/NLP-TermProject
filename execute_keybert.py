import json
from keybert import KeyBERT

# 🔧 파일 경로 설정
INPUT_JSON = "top_clusters_for_keyword.json"              # 클러스터별 논문 정보 입력 파일
OUTPUT_JSON = "top_clusters_after_bert_with_keyword.json" # 키워드를 추가한 출력 파일
TOP_K_KEYWORDS = 5                                        # 클러스터별 추출할 키워드 수

# 🔹 1. JSON 파일 로드
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# 상위 클러스터 ID 및 논문 리스트 정보 추출
top_clusters = data["top_clusters"]              # 상위 PageRank 클러스터 ID 리스트
cluster_details = data["cluster_details"]        # 클러스터별 논문 ID 및 제목 목록

# 🔹 2. KeyBERT 모델 로드 (기본 모델: all-MiniLM-L6-v2)
kw_model = KeyBERT("all-MiniLM-L6-v2")

# 🔹 3. 클러스터별 키워드 추출 시작
cluster_keywords = {}

for cluster_id in top_clusters:
    papers = cluster_details[str(cluster_id)]  # 클러스터 ID는 문자열 키로 접근
    titles = [paper["title"] for paper in papers]  # 논문 제목만 추출

    # 제목들을 공백으로 연결하여 하나의 긴 문서처럼 구성
    text = " ".join(titles)

    # KeyBERT로 키워드 추출 (중복 제거 + 중요도 순 상위 N개)
    keywords = kw_model.extract_keywords(
        text,
        top_n=TOP_K_KEYWORDS,
        stop_words='english'   # 불용어 제거
    )
    keywords = [kw for kw, _ in keywords]  # 키워드 텍스트만 추출

    # 결과 저장
    cluster_keywords[cluster_id] = keywords
    print(f"📌 Cluster {cluster_id} → {keywords}")

# 🔹 4. 추출한 키워드를 원본 데이터에 추가 후 저장
data["cluster_keywords"] = cluster_keywords

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 키워드 저장 완료: {OUTPUT_JSON}")
