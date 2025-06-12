import pandas as pd
import logging
import os

# --- Configuration ---
# 1. Path to your main dataset that has missing embeddings.
DATASET_TO_REPAIR = "repaired_dataset_with_filled_embeddings.csv"

# 2. Path to the CSV file containing the substitution plan.
SUBSTITUTES_FILE = "embedding_substitutes.csv"

# 3. Path for the new, repaired output file.
OUTPUT_FILE = "repaired_dataset_with_filled_embeddings.csv"


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def batch_fill_embeddings(main_file_path, substitutes_file_path, output_path):
    """
    Automates filling missing embeddings based on a substitute mapping file.
    """
    if not os.path.exists(main_file_path):
        logging.error(f"Main dataset file not found: '{main_file_path}'")
        return
    if not os.path.exists(substitutes_file_path):
        logging.error(f"Substitutes mapping file not found: '{substitutes_file_path}'")
        return

    logging.info(f"Loading main dataset from '{main_file_path}'...")
    main_df = pd.read_csv(main_file_path)

    logging.info(f"Loading substitutes mapping from '{substitutes_file_path}'...")
    substitutes_df = pd.read_csv(substitutes_file_path)

    # --- Step 1: Create an embedding bank for fast lookup ---
    embedding_bank = {
        row['paperId']: row['embedding']
        for index, row in main_df.iterrows()
        if pd.notna(row['embedding']) and str(row['embedding']).strip() not in ['null', '[]']
    }
    logging.info(f"Created an embedding bank with {len(embedding_bank)} valid embeddings.")

    # --- Step 2: Iterate through the substitution plan ---
    fill_count = 0
    
    # --- FIX: Using the standard and reliable iterrows() method ---
    for index, sub_row in substitutes_df.iterrows():
        # Access columns by their exact string name, which handles spaces.
        target_id = sub_row["Target Paper ID"]
        substitute_id = sub_row["Substitute Paper ID"]

        if pd.isna(target_id):
            logging.warning(f"Skipping row {index+1} in substitutes file due to missing Target Paper ID.")
            continue

        target_index_list = main_df.index[main_df['paperId'] == target_id].tolist()
        if not target_index_list:
            logging.warning(f"Target paper {target_id} not found in main dataset. Skipping.")
            continue
        
        target_index = target_index_list[0]

        if pd.notna(main_df.at[target_index, 'embedding']) and str(main_df.at[target_index, 'embedding']).strip() not in ['null', '[]']:
            logging.info(f"Target paper {target_id} already has an embedding. Skipping.")
            continue

        # --- Step 3: Find a valid substitute embedding ---
        source_embedding = embedding_bank.get(substitute_id)

        if source_embedding:
            main_df.at[target_index, 'embedding'] = source_embedding
            logging.info(f"Filled embedding for {target_id} using substitute {substitute_id}")
            fill_count += 1
        else:
            logging.error(f"Could not find a valid embedding for substitute paper {substitute_id}. Cannot fill target {target_id}.")

    logging.info(f"Finished processing. Filled a total of {fill_count} missing embeddings.")

    # --- Step 4: Save the repaired dataset ---
    try:
        main_df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved repaired dataset to '{output_path}'.")
    except IOError as e:
        logging.error(f"Failed to save output file: {e}")


if __name__ == "__main__":
    batch_fill_embeddings(DATASET_TO_REPAIR, SUBSTITUTES_FILE, OUTPUT_FILE)