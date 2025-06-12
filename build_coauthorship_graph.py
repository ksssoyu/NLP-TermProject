"""
Build Co-Authorship Graph from Paper Metadata

This script processes academic paper metadata in CSV format to construct a weighted undirected co-authorship graph.
Each author is represented as a node, and each co-authorship relationship as an edge. Edge weights indicate the
number of collaborations, and edge attributes include the years of collaboration.

Main Features:
- Reads paper metadata from a CSV file containing author lists and publication years.
- Builds a NetworkX undirected graph where:
    - Nodes are authors
    - Edges represent co-authorship (with weights and collaboration years)
- Stores edge years as comma-separated strings for compatibility with GraphML format.
- Exports the resulting graph to a GraphML file for further analysis or visualization.

Input:
- v2_papers.csv: CSV file with columns including 'authors' (as a JSON string) and 'year'.

Output:
- coauthorship_graph_with_year.graphml: GraphML file representing the co-authorship network.

Usage:
- Run directly as a script: `python build_coauthorship_graph.py`
"""


import pandas as pd
import networkx as nx
import json
import logging
from tqdm import tqdm
from itertools import combinations

# --- Configuration ---
INPUT_CSV_FILE = "v2_papers.csv"
OUTPUT_GRAPH_FILE = "coauthorship_graph_with_year.graphml"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main function ---

def build_coauthorship_graph(csv_path):
    """
    Build an author co-authorship graph from paper data.
    """
    logging.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path).where(pd.notna, None)
        if df.empty:
            logging.warning(f"Input CSV file '{csv_path}' is empty.")
            return nx.Graph()
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}.")
        return None

    logging.info(f"Loaded {len(df)} papers. Building co-authorship graph...")

    G = nx.Graph()

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing papers"):
        authors_raw = row.get('authors')
        year = row.get('year')  # 연도 정보 가져오기
        if not authors_raw or not year:
            continue
        try:
            authors = json.loads(authors_raw)
            if not isinstance(authors, list):
                continue
        except (json.JSONDecodeError, TypeError):
            continue

        # Clean author names (optional)
        authors = [author.strip() for author in authors if author]

        # Add nodes
        for author in authors:
            if not G.has_node(author):
                G.add_node(author)

        # Add edges (pairwise combinations) with year information
        for a1, a2 in combinations(authors, 2):
            if G.has_edge(a1, a2):
                # 엣지에 이미 존재하면 연도 정보를 추가로 기록 (기존 연도와 중복되지 않게 추가)
                if year not in G[a1][a2]['years']:
                    G[a1][a2]['years'].append(year)
                G[a1][a2]['weight'] += 1
            else:
                # 새로운 엣지 추가 시, 연도 정보와 weight 추가
                G.add_edge(a1, a2, weight=1, years=[year])

    logging.info("Graph construction complete.")
    logging.info(f"Number of authors (nodes): {G.number_of_nodes()}")
    logging.info(f"Number of co-authorship relations (edges): {G.number_of_edges()}")
    return G

def convert_edge_years_to_string(G):
    """
    Convert edge 'years' list to a comma-separated string for compatibility with GraphML.
    """
    for u, v, data in G.edges(data=True):
        if 'years' in data:
            data['years'] = ','.join(map(str, data['years']))  # Convert list to comma-separated string
    return G

# --- Main Execution ---
if __name__ == "__main__":
    coauthor_graph = build_coauthorship_graph(INPUT_CSV_FILE)

    if coauthor_graph is not None:
        num_nodes = coauthor_graph.number_of_nodes()
        num_edges = coauthor_graph.number_of_edges()
        isolated_nodes = len(list(nx.isolates(coauthor_graph)))

        print("\n" + "="*40)
        print("Co-authorship Graph Summary")
        print("="*40)
        print(f"Number of authors (nodes): {num_nodes}")
        print(f"Number of co-authorship edges: {num_edges}")
        print(f"Number of isolated authors: {isolated_nodes}")
        print("="*40)

        try:
            # Convert edge years to string format before saving
            coauthor_graph = convert_edge_years_to_string(coauthor_graph)

            logging.info(f"Saving co-authorship graph to {OUTPUT_GRAPH_FILE}...")
            nx.write_graphml(coauthor_graph, OUTPUT_GRAPH_FILE)
            logging.info("Graph saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save graph: {e}")
    else:
        logging.error("Graph building failed.")
