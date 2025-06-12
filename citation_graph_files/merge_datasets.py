import pandas as pd
import logging

# --- Configuration ---
# Your original dataset with superstar papers
EXISTING_DATASET_PATH = "nlp_papers_dataset_v6_with_isInfluential.csv"
# The new dataset fetched by the script above
NEW_DATASET_PATH = "nlp_papers_2425_top_venues.csv"
# The final, combined output file
MERGED_OUTPUT_PATH = "cut5.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_and_deduplicate():
    """
    Merges the existing and new datasets, removing duplicates based on paperId.
    """
    logging.info(f"Loading existing dataset from: {EXISTING_DATASET_PATH}")
    try:
        df_existing = pd.read_csv(EXISTING_DATASET_PATH)
    except FileNotFoundError:
        logging.error(f"File not found: {EXISTING_DATASET_PATH}. Cannot proceed.")
        return

    logging.info(f"Loading new dataset from: {NEW_DATASET_PATH}")
    try:
        df_new = pd.read_csv(NEW_DATASET_PATH)
    except FileNotFoundError:
        logging.error(f"File not found: {NEW_DATASET_PATH}. Cannot proceed.")
        return

    logging.info(f"Existing dataset has {len(df_existing)} papers.")
    logging.info(f"New dataset has {len(df_new)} papers.")

    # --- Combine and Deduplicate ---
    # Append the new dataframe to the existing one
    combined_df = pd.concat([df_existing, df_new], ignore_index=True)

    # Remove duplicates based on 'paperId', keeping the first occurrence.
    # Since the existing data is first, its entries will be preserved.
    logging.info(f"Total papers before deduplication: {len(combined_df)}")
    merged_df = combined_df.drop_duplicates(subset='paperId', keep='first')
    logging.info(f"Total papers after deduplication: {len(merged_df)}")

    # --- Save the final dataset ---
    try:
        # Re-align columns to match the original schema as much as possible
        # This brings columns from the original dataset to the front
        final_columns = [col for col in df_existing.columns if col in merged_df.columns]
        # Add any new columns from the new dataset that weren't in the old one
        for col in df_new.columns:
            if col not in final_columns:
                final_columns.append(col)
        
        final_df = merged_df[final_columns]

        final_df.to_csv(MERGED_OUTPUT_PATH, index=False)
        logging.info(f"Successfully saved merged dataset to {MERGED_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save merged file: {e}")

if __name__ == "__main__":
    merge_and_deduplicate()