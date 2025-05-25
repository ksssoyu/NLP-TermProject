from analyze_graph import CitationGraphAnalyzer 

def main():
    analyzer = CitationGraphAnalyzer("nlp_citation_graph.gpickle")
    analyzer.run_louvain_clustering()
    analyzer.run_pagerank()
    analyzer.get_top_papers_by_pagerank(top_n=10)
    analyzer.get_top_concepts_by_cluster(top_n=10)
    analyzer.save_pagerank_to_csv()

if __name__ == "__main__":
    main()