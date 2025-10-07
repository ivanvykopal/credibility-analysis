from src.eval import calculate_retrieval_metrics, calculate_retrieval_metrics_all

if __name__ == "__main__":
    
    sources = [] # This is only for a single evaluation with a list of sources
    
    print(calculate_retrieval_metrics(sources, ignore_missing=True))
    
    sources = [[]] # This is overall evaluation across all examples with a list of lists of sources
    
    print(calculate_retrieval_metrics_all(sources, ignore_missing=True))
    