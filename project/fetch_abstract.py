import pandas as pd
import requests
import time
from tqdm import tqdm

# ✅ 설정
INPUT_CSV = "merged_nlp_papers_for_enrichment.csv"
OUTPUT_CSV = "merged_nlp_papers_with_abstract.csv"
FAILED_CSV = "abstract_failed_ids.csv"
API_KEY = "ZwAf0KO0DJ82Z87lH5ODz8xvto6uk36uaoz2dUy5"
HEADERS = {"x-api-key": API_KEY}
API_URL = "https://api.semanticscholar.org/graph/v1/paper/"
FIELDS = "paperId,title,abstract"
MAX_RETRIES = 6

# ✅ 데이터 로드
df = pd.read_csv(INPUT_CSV)
paper_ids = df['paperId'].tolist()
abstracts = [""] * len(paper_ids)

# ✅ 매핑 구조
id_to_index = {pid: idx for idx, pid in enumerate(paper_ids)}
retry_counts = {pid: 0 for pid in paper_ids}
failed_ids = []
give_up_ids = []

# ✅ 1차 시도
print("🚀 1차 시도 중...")
for pid in tqdm(paper_ids):
    idx = id_to_index[pid]
    try:
        response = requests.get(f"{API_URL}{pid}", headers=HEADERS, params={"fields": FIELDS})
        if response.status_code == 200:
            data = response.json()
            abstracts[idx] = data.get("abstract", "")
        else:
            print(f"❌ 실패: {pid} (status: {response.status_code})")
            failed_ids.append(pid)
    except Exception as e:
        print(f"⚠️ 예외 발생: {pid} (error: {e})")
        failed_ids.append(pid)
    time.sleep(0.7)

# ✅ 반복 재시도
attempt = 1
while failed_ids:
    print(f"\n🔁 재시도 {attempt}차 (남은 개수: {len(failed_ids)})...")
    new_failed = []
    for pid in tqdm(failed_ids):
        idx = id_to_index[pid]
        try:
            response = requests.get(f"{API_URL}{pid}", headers=HEADERS, params={"fields": FIELDS})
            if response.status_code == 200:
                data = response.json()
                abstracts[idx] = data.get("abstract", "")
            else:
                print(f"❌ 실패: {pid} (status: {response.status_code})")
                retry_counts[pid] += 1
                if retry_counts[pid] < MAX_RETRIES:
                    new_failed.append(pid)
                else:
                    print(f"❗️ 최대 재시도 초과: {pid}, 포기")
                    give_up_ids.append(pid)
        except Exception as e:
            print(f"⚠️ 예외 발생: {pid} (error: {e})")
            retry_counts[pid] += 1
            if retry_counts[pid] < MAX_RETRIES:
                new_failed.append(pid)
            else:
                print(f"❗️ 최대 재시도 초과: {pid}, 포기")
                give_up_ids.append(pid)
        time.sleep(0.9)
    failed_ids = new_failed
    attempt += 1

# ✅ 저장
df["abstract"] = abstracts
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ abstract 수집 완료 및 저장: {OUTPUT_CSV}")

# ✅ 실패 ID 출력 및 저장
if give_up_ids:
    print(f"\n🚫 최종적으로 abstract를 받아오지 못한 논문 수: {len(give_up_ids)}")
    print("🧾 실패 논문 ID 목록:")
    for pid in give_up_ids:
        print(f" - {pid}")
    pd.DataFrame({"failed_paperId": give_up_ids}).to_csv(FAILED_CSV, index=False)
    print(f"📁 실패 논문 ID가 저장됨: {FAILED_CSV}")
else:
    print("\n🎉 모든 논문의 abstract를 성공적으로 수집했습니다!")
