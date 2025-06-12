import pandas as pd
import logging
import os

# --- Configuration ---
# 1. Set the path to the CSV file you want to analyze.
#    This should be your most complete dataset.
INPUT_CSV_FILE = "final_v2_papers.csv"

# 2. Set how many of the top papers you want to display.
TOP_N_TO_SHOW = 500

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_top_papers_with_embeddings(file_path, top_n):
    """
    Loads a CSV file, finds papers WITH embeddings, sorts them by citation count,
    and prints the top N.
    """
    if not os.path.exists(file_path):
        logging.error(f"Error: File not found at '{file_path}'")
        return

    logging.info(f"Loading data from '{file_path}' to find top-cited papers with embeddings...")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file. Error: {e}")
        return

    # --- Step 1: Filter to keep ONLY papers that HAVE a valid embedding ---
    # A valid embedding is not null and is not an empty JSON array string.
    valid_embedding_mask = (df['embedding'].notnull()) & (df['embedding'] != 'null') & (df['embedding'] != '[]')
    papers_with_embeddings = df[valid_embedding_mask].copy()

    if not papers_with_embeddings.empty:
        # --- NEW: Sort by citationCount and get the top N ---
        logging.info("Sorting papers by citation count...")
        # Ensure citationCount is numeric before sorting
        papers_with_embeddings['citationCount'] = pd.to_numeric(papers_with_embeddings['citationCount'], errors='coerce').fillna(0)
        
        top_missing_papers = papers_with_embeddings.sort_values(
            by='citationCount', ascending=False
        ).head(100)

        count = len(top_missing_papers)
        logging.warning(f"Displaying the Top {count} paper(s) with missing embedding values (sorted by citation count):")
        
        # Iterate through the top papers and print their details
        for row in top_missing_papers.itertuples(index=False):
            paper_id = row.paperId
            title = row.title
            citations = int(row.citationCount)
            print(f"  Citations: {citations:<5} | ID: {paper_id} | Title: {title}")
            
        logging.info(f"--- End of list ---")
    else:
        logging.info("No papers with missing embeddings were found in the file.")


    if papers_with_embeddings.empty:
        logging.warning("No papers with valid embeddings were found in the file.")
        return

    logging.info(f"Found {len(papers_with_embeddings)} papers with embeddings. Sorting by citation count...")

    # --- Step 2: Sort the remaining papers by citation count ---
    papers_with_embeddings['citationCount'] = pd.to_numeric(papers_with_embeddings['citationCount'], errors='coerce').fillna(0)
    papers_with_embeddings['citationCount'] = papers_with_embeddings['citationCount'].astype(int)

    top_papers_df = papers_with_embeddings.sort_values(by='citationCount', ascending=False).head(top_n)
    # --- Step 3: Print the results ---
    print("\n" + "="*80)
    print(f"Top {len(top_papers_df)} Most-Cited Papers WITH EMBEDDINGS (Candidates for Substitution)")
    print("="*80)
    
    for row in top_papers_df.itertuples(index=False):
        paper_id = row.paperId
        title = row.title
        citations = row.citationCount
        print(f"Citations: {citations:<7} | ID: {paper_id} | Title: {title}")
        
    print("="*80)


if __name__ == "__main__":
    display_top_papers_with_embeddings(INPUT_CSV_FILE, TOP_N_TO_SHOW)