import pandas as pd
import logging
import os

# --- Configuration ---
# Set the path to the CSV file you want to check.
INPUT_CSV_FILE = "final_v2_papers.csv"
# Set how many of the top papers to display
TOP_N_TO_DISPLAY = 100

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_top_missing_embeddings(file_path):
    """
    Loads a CSV, finds papers with missing embeddings, sorts them by citation count,
    and prints the top N.
    """
    if not os.path.exists(file_path):
        logging.error(f"Error: File not found at '{file_path}'")
        return

    logging.info(f"Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file. Error: {e}")
        return

    # --- Find rows with missing embeddings ---
    missing_mask = (df['embedding'].isnull()) | (df['embedding'] == 'null') | (df['embedding'] == '[]')
    papers_with_missing_embeddings = df[missing_mask].copy()

    if not papers_with_missing_embeddings.empty:
        # --- NEW: Sort by citationCount and get the top N ---
        logging.info("Sorting papers by citation count...")
        # Ensure citationCount is numeric before sorting
        papers_with_missing_embeddings['citationCount'] = pd.to_numeric(papers_with_missing_embeddings['citationCount'], errors='coerce').fillna(0)
        
        top_missing_papers = papers_with_missing_embeddings.sort_values(
            by='citationCount', ascending=False
        ).head(TOP_N_TO_DISPLAY)

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

if __name__ == "__main__":
    find_top_missing_embeddings(INPUT_CSV_FILE)