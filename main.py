from analyze_graph import CitationGraphAnalyzer

def main():
    analyzer = CitationGraphAnalyzer("nlp_citation_graph.gpickle")

    # cluster and take top 10 most common concepts in each cluster
    analyzer.run_louvain_clustering()
    analyzer.save_top_concepts_by_cluster(top_n=10)

    # run pagerank and eigenvector centrality
    analyzer.run_pagerank()
    analyzer.run_eigenvector_centrality()

    # just top ten of each. full ranking will be stored in csv
    pagerank_result : list[tuple[str, float]] = analyzer.get_top_papers_by_pagerank(top_n=10)
    eigen_cent_result : list[tuple[str, float]] = analyzer.get_top_papers_by_eigen_centr(top_n=10)

    # quick top10 check
    for i, pagerank in enumerate(pagerank_result, 1):
        print(f"{i} {pagerank[0]} : {pagerank[1]}")
    
    print()
    for i, paper_entry in enumerate(eigen_cent_result, 1):
        print(f"{i} {paper_entry[0]} : {paper_entry[1]}")

    analyzer.save_pagerank_to_csv("pagerank_rankings.csv")
    analyzer.save_eigenvector_to_csv("eigenvector_rankings.csv")

    analyzer.analyze_per_cluster()

if __name__ == "__main__":
    main()