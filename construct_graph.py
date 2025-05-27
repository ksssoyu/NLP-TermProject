print("\n--- Starting Graph Construction ---")
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm

df_nlp_papers = pd.read_csv("nlp_papers.csv")

G = nx.DiGraph()

# Add nodes with attributes from the filtered NLP papers
for index, row in tqdm(df_nlp_papers.iterrows(), total=len(df_nlp_papers), desc="Adding nodes"):
    paper_id = row['id']
    G.add_node(paper_id,
               title=row.get('title'),
               year=row.get('publication_year'),
               authors=row.get('authors'),
               concepts=row.get('concepts'),
               citations_count=row.get('citations'))

print(f"Added {G.number_of_nodes()} nodes (NLP papers) to the graph.")

# Add edges (citations)
# We only add edges between papers that are *both* in our filtered NLP set.
# This creates a "subgraph" of only NLP-relevant citation relationships.
for index, row in tqdm(df_nlp_papers.iterrows(), total=len(df_nlp_papers), desc="Adding edges"):
    citing_paper_id = row['id']
    # Split the semicolon-separated string of referenced works
    referenced_ids_str = str(row.get('referenced_works', '')) # Ensure it's a string
    if referenced_ids_str: # Check if string is not empty
        # Split by semicolon and clean each ID
        cited_paper_ids = [ref.strip() for ref in referenced_ids_str.split(';') if ref.strip()]
        for cited_id in cited_paper_ids:
            # Add edge ONLY if the cited paper is also a node in our current NLP graph
            if G.has_node(citing_paper_id) and G.has_node(cited_id):
                G.add_edge(citing_paper_id, cited_id) # Edge from citing paper to cited paper

print(f"Added {G.number_of_edges()} edges (citations) to the graph.")
with open("nlp_citation_graph.gpickle", "wb") as f:
    pickle.dump(G, f)
