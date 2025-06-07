import networkx as nx
import logging
import ast # Used for safely evaluating string-formatted lists
from collections import Counter

# --- Configuration ---
# The path to the graph file created by the graph builder script.
INPUT_GRAPH_FILE = "citation_graph.graphml"

# --- Relevance Criteria (mirrors the logic from the crawler) ---
# These lists help us categorize the papers in the graph.
# You can adjust these if you want to refine the validation criteria.

# Papers with these S2 Fields of Study are considered "Core NLP".
CORE_S2_NLP_FOS = ['linguistics', 'information retrieval']

# Papers published in these venues are considered "Top AI/NLP Venue".
# This list should match the one used in your crawler.
TOP_NLP_VENUES = [
    'acl', 'emnlp', 'naacl', 'coling', 'eacl',
    'transactions of the association for computational linguistics', 'computational linguistics', 'lrec',
    'neurips', 'icml', 'iclr', 'aaai', 'ijcai'
]

# The general category that was also used for inclusion.
CS_FOS_CATEGORY = "computer science"


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Analysis Function ---

def validate_dataset_composition(graph_path):
    """
    Loads a graph and analyzes the attributes of its nodes to validate
    their relevance to NLP.
    
    Args:
        graph_path (str): The path to the input .graphml file.
    """
    logging.info(f"Loading graph from {graph_path} for validation...")
    try:
        G = nx.read_graphml(graph_path)
    except FileNotFoundError:
        logging.error(f"File not found: {graph_path}. Please run the graph builder script first.")
        return

    if not G.nodes():
        logging.warning("Graph is empty. Cannot perform validation.")
        return

    logging.info(f"Analyzing {G.number_of_nodes()} papers in the graph...")

    # --- Categorize each paper ---
    # We will assign each paper to the 'strongest' category it qualifies for.
    category_counts = Counter()

    for node_id, data in G.nodes(data=True):
        # The attributes were saved as strings in GraphML, so we need to process them.
        s2_fields_str = data.get('s2FieldsOfStudy', '[]')
        venue_name = data.get('publicationVenueName', '').lower()
        
        # Safely parse the string representation of the list
        try:
            # ast.literal_eval is safer than eval() for parsing Python literals
            s2_fields_list = ast.literal_eval(s2_fields_str)
            # Ensure it's a list of lowercase strings for consistent checking
            s2_fields_list = [str(f).lower() for f in s2_fields_list]
        except (ValueError, SyntaxError):
            logging.warning(f"Could not parse s2FieldsOfStudy for paper {node_id}: {s2_fields_str}")
            s2_fields_list = []
        
        # Check categories in order of specificity (strongest signals first)
        
        # 1. Check for Core NLP Fields of Study
        is_core_nlp = any(core_fos in s2_fields_list for core_fos in CORE_S2_NLP_FOS)
        if is_core_nlp:
            category_counts['Core NLP (by S2 Field)'] += 1
            continue # Move to the next paper once categorized

        # 2. Check for Top AI/NLP Venues
        is_top_venue = any(top_venue in venue_name for top_venue in TOP_NLP_VENUES)
        if is_top_venue:
            category_counts['Top AI/NLP Venue'] += 1
            continue

        # 3. Check if it was likely included because of the "Computer Science" rule
        if CS_FOS_CATEGORY in s2_fields_list:
            category_counts['General Computer Science'] += 1
            continue
            
        # 4. If none of the above, categorize as 'Other'
        category_counts['Other'] += 1
        
    # --- Print the validation report ---
    total_papers = G.number_of_nodes()
    
    print("\n" + "="*80)
    print("Dataset Composition Validation Report")
    print("="*80)
    print(f"Total papers analyzed: {total_papers}\n")
    
    print(f"{'Category':<35} | {'Count':>10} | {'Percentage':>15}")
    print("-"*80)

    # Calculate and print stats for each category
    for category, count in category_counts.most_common():
        percentage = (count / total_papers) * 100 if total_papers > 0 else 0
        print(f"{category:<35} | {count:>10,} | {f'{percentage:14.2f}%'}")
    
    print("-"*80)
    
    # Calculate a "Pure NLP" metric
    pure_nlp_count = category_counts.get('Core NLP (by S2 Field)', 0) + category_counts.get('Top AI/NLP Venue', 0)
    pure_nlp_percentage = (pure_nlp_count / total_papers) * 100 if total_papers > 0 else 0
    
    print(f"\nSummary Metric:")
    print(f"  - Papers classified as 'Core NLP' or from a 'Top AI/NLP Venue': {pure_nlp_count:,} ({pure_nlp_percentage:.2f}%)")
    print("="*80)

# --- Main Execution ---
if __name__ == "__main__":
    validate_dataset_composition(INPUT_GRAPH_FILE)
