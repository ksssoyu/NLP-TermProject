import pandas as pd
import json

# --- CONFIGURATION ---
INPUT_CSV_FILE = "final_v2_papers.csv"
# IMPORTANT: Put the ID of one of the missing top papers here
PAPER_ID_TO_CHECK = "204e3073870fae3d05bcbc2f6a8e263d9b72e776" 

print(f"--- Diagnosing Paper ID: {PAPER_ID_TO_CHECK} ---")

# 1. Load the data
try:
    df = pd.read_csv(INPUT_CSV_FILE)
except FileNotFoundError:
    print(f"\n[X] ERROR: CSV file not found at '{INPUT_CSV_FILE}'")
    exit()

# 2. Find the specific paper before any filtering
paper_row = df[df['paperId'] == PAPER_ID_TO_CHECK]

if paper_row.empty:
    print(f"\n[X] CRITICAL ERROR: Paper ID '{PAPER_ID_TO_CHECK}' was not found in the original CSV file.")
    exit()

print("\n[STEP 1: RAW DATA]")
embedding_value = paper_row['embedding'].iloc[0]
print(f"Raw 'embedding' value: {embedding_value}")
print(f"Type of raw value: {type(embedding_value)}")

# 3. Simulate the first cleaning step: dropna()
df_after_dropna = df.dropna(subset=['embedding'])
if PAPER_ID_TO_CHECK not in df_after_dropna['paperId'].values:
    print("\n[X] RESULT: Paper was REMOVED by df.dropna(subset=['embedding'])")
    print("This means pandas read the embedding as a null/NaN value.")
    exit()
else:
    print("\n[✓] PASSED: Paper survived the dropna() filter.")

# 4. Simulate the second cleaning step: isin(['', 'null', '[]'])
df_after_isin = df_after_dropna[df_after_dropna['embedding'].str.strip().isin(['', 'null', '[]']) == False]
if PAPER_ID_TO_CHECK not in df_after_isin['paperId'].values:
    print(f"\n[X] RESULT: Paper was REMOVED by the check for empty strings or '[]'.")
    print(f"Its value was '{str(embedding_value).strip()}', which was caught by the filter.")
    exit()
else:
    print("[✓] PASSED: Paper survived the empty string filter.")

# 5. Simulate the third cleaning step: json.loads()
print("\n[STEP 2: JSON PARSING TEST]")
try:
    json.loads(embedding_value)
    print("[✓] PASSED: The embedding string is valid JSON.")
except Exception as e:
    print(f"\n[X] RESULT: The script would FAIL at the json.loads() step.")
    print(f"ERROR: {e}")
    print("This is a critical error. The embedding string is not formatted correctly.")
    exit()

print("\n--- CONCLUSION ---")
print("The paper seems to pass all data cleaning and formatting checks.")
print("This suggests the problem is Reason #2: The paper is likely in a small cluster that is not being displayed in the final report.")