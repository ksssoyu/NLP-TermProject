import pandas as pd
import logging

# --- Configuration ---
# The path to the dataset CSV file created by the crawler.
# Make sure this matches the output file from your data collection script.
INPUT_CSV_FILE = "nlp_papers_dataset_v6_with_isInfluential.csv"
# The number of top-ranking papers to display.
TOP_N = 500

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Analysis Function ---

def analyze_citation_counts(csv_path):
    """
    Loads paper data from a CSV, calculates the average citation count,
    and displays the papers with the highest citation counts.
    
    Args:
        csv_path (str): The path to the input CSV file.
    """
    logging.info(f"Loading data from {csv_path}...")
    try:
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.warning(f"The CSV file '{csv_path}' is empty. No analysis can be performed.")
            return
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}. Please ensure the data collection script ran successfully.")
        return
    except pd.errors.EmptyDataError:
        logging.warning(f"The CSV file '{csv_path}' contains no data. No analysis can be performed.")
        return

    # --- 1. Calculate Average Citation Count ---
    # Ensure the 'citationCount' column is numeric, converting errors to NaN, then filling with 0
    df['citationCount'] = pd.to_numeric(df['citationCount'], errors='coerce').fillna(0)
    
    average_citations = df['citationCount'].mean()
    
    logging.info("Calculation complete.")
    print("\n" + "="*80)
    print("Dataset Citation Statistics")
    print("="*80)
    print(f"Average citation count across all {len(df)} collected papers: {average_citations:.2f}")
    print("="*80)


    # --- 2. Order by Citation Count and Display Top N ---
    # Sort the DataFrame by the 'citationCount' column in descending order
    sorted_df = df.sort_values(by='citationCount', ascending=False)
    
    print(f"Top {TOP_N} Papers by Raw Citation Count")
    print("="*80)
    print(f"{'Rank':<5} | {'Citations':<10} | {'Year':<6} | {'Title'}")
    print("-"*80)

    # Display the top N papers
    for i, row in enumerate(sorted_df.head(TOP_N).itertuples()):
        rank = i + 1
        citations = int(row.citationCount)
        year = int(row.year) if pd.notna(row.year) else 'N/A'
        title = str(row.title)
        
        # Truncate long titles for display
        display_title = (title[:70] + '...') if len(title) > 70 else title
        
        print(f"{rank:<5} | {citations:<10,} | {str(year):<6} | {display_title}")
        
    print("="*80)


# --- Main Execution ---
if __name__ == "__main__":
    analyze_citation_counts(INPUT_CSV_FILE)
