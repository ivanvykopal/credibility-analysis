import pandas as pd
from typing import List
from src.sites_mapping import SITES_MAPPING, EDMO, SITES_CATEGORIES


def url_to_domain(url: str) -> str:
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    
    if parsed_url.netloc.startswith('www.'):
        parsed_url = parsed_url._replace(netloc=parsed_url.netloc[4:])
        
    # handle also subdomains by returning only the main domain
    if parsed_url.netloc:
        domain_parts = parsed_url.netloc.split('.')
        if len(domain_parts) > 2:
            parsed_url = parsed_url._replace(netloc='.'.join(domain_parts[-2:]))
            
        return parsed_url.netloc.lower()
    else:
        return url.lower()


FACTUALITY_MAPPING = {
    'satire': -1,
    'very low': -1,
    'low': -0.5,
    'mixed': 0,
    'not rated': 0,
    'mostly factual': 0.5,
    'high': 1,
    'very high': 1,
}

POLITICAL_BIAS_MAPPING = {
    'extreme right': -1,
    'far right': -1,
    'right conspiracy': -1,
    'right pseudoscience': -1,
    'right conspiracy pseudoscience': -1,
    'far right conspiracy': -1,
    'extreme right conspiracy': -1,
    'alt-right conspiracy': -1,
    'right satire': -0.5,
    'right': -0.5,
    'right-center': -0.5,
    'right center': -0.5,
    'center': 0,
    'least biased': 0,
    'pro': 0,
    'least - pro science': 0,
    'not rated': 0,
    'left leaning pro': 0.5,
    'left-center': 0.5,
    'left': 0.5,
    'left satire': 0.5,
    'left conspiracy': 1,
    'left pseudoscience': 1,
    'far left': 1,
    'extreme left': 1,
    'conspiracy': 0,
    'pseudoscience': 0,
    'mild pseudoscience': 0,
    'quackery pseudscience': 0,
    'junk news': 0,
    'satire': 0,
}

def load_fcs() -> pd.DataFrame:
    df = pd.read_csv('data/fact-checking-organizations.csv')
    df['normalized_domain'] = df['url'].apply(lambda x: url_to_domain(x))
    
    # Add EDMO domains with high factuality
    edmo_domains = [url_to_domain(url) for url in EDMO]
    df = pd.concat([
        df,
        pd.DataFrame({'url': EDMO, 'normalized_domain': edmo_domains})
    ], ignore_index=True)
    
    return df


def load_mbfc() -> pd.DataFrame:
    df = pd.read_csv('data/mbfc.csv')
    df['normalized_domain'] = df['domain'].apply(lambda x: url_to_domain(x) if pd.notna(x) else None)
    
    for original, existing in SITES_MAPPING.items():
        metadata = df[df['normalized_domain'] == existing]
        metadata.loc[:, 'normalized_domain'] = original
        df = pd.concat([df, metadata], ignore_index=True)
    
    return df


def load_factuality_data() -> dict:
    fcs = load_fcs()
    mbfc = load_mbfc()
    
    fcs_dict = {
        **{row['normalized_domain']: 'HIGH' for _, row in fcs.iterrows()},
    }
    
    mbfc_dict = {
        **{row['normalized_domain']: row['factuality_rating'] for _, row in mbfc.iterrows() if pd.notna(row['factuality_rating'])},
    }
    
    for domain, category in SITES_CATEGORIES.items():
        if category in ['government', 'fact-checking', 'publications', 'reliable-news']:
            fcs_dict[url_to_domain(domain)] = 'HIGH'
        if category in ['social-media']:
            fcs_dict[url_to_domain(domain)] = 'MIXED'
        if category in ['disinformation']:
            fcs_dict[url_to_domain(domain)] = 'VERY LOW'
    
    factuality_data = {**fcs_dict, **mbfc_dict}
    factuality_data = {k: v for k, v in factuality_data.items() if v is not None}
    return factuality_data


def load_bias_data() -> dict:
    mbfc = load_mbfc()
    
    mbfc_dict = {
        **{row['normalized_domain']: row['bias_rating'] for _, row in mbfc.iterrows() if pd.notna(row['bias_rating'])},
    }
    
    return mbfc_dict


def calculate_scs(sources: List[str], ignore_missing=False) -> float:
    """
    Calculating Source Credibility Score
    
    :param sources: List of sources (URLs)
    :return: Source Factuality Score in the range -1 to 1
    :rtype: float
    
    This function calculates the Source Factuality Score (SFS) based on the sources provided.
    """
    factuality_data = load_factuality_data()
    
    if ignore_missing:
        sources = [source for source in sources if url_to_domain(source) in factuality_data]
    
    scores = [
        FACTUALITY_MAPPING.get(factuality_data.get(url_to_domain(source), 'NOT RATED').lower(), 0)
        for source in sources
    ]
    
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
    

def overall_credibility_score(sources: List[List[str]], ignore_missing=False) -> float:
    """
    Calculate the overall credibility score for a list of sources.
    
    :param sources: List of lists of sources (URLs)
    :return: Overall factuality score in the range -1 to 1
    :rtype: float
    
    This function calculates the overall factuality score based on the Source Factuality Score (SFS) for each source.
    """
    sfs_scores = [calculate_scs(source, ignore_missing) for source in sources]
    
    if not sfs_scores:
        return 0.0
    return sum(sfs_scores) / len(sfs_scores)


def calculate_spbs(sources: List[str], ignore_missing=False) -> float:
    """
    Calculating Source Political Bias Score
    
    :param sources: List of sources (URLs)
    :return: Source Political Bias Score in the range -1 to 1
    :rtype: float
    
    This function calculates the Source Political Bias Score (SPBS) based on the sources provided.
    """
    bias_data = load_bias_data()

    
    if ignore_missing:
        sources = [source for source in sources if url_to_domain(source) in bias_data]
    
    scores = [
        POLITICAL_BIAS_MAPPING.get(bias_data.get(url_to_domain(source), 'NOT RATED').lower(), 0)
        for source in sources
    ]
    
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def overall_political_bias_score(sources: List[List[str]], ignore_missing=False) -> float:
    """
    Calculate the overall political bias score for a list of sources.
    
    :param sources: List of lists of sources (URLs)
    :return: Overall political bias score in the range -1 to 1
    :rtype: float
    
    This function calculates the overall political bias score based on the Source Political Bias Score (SPBS) for each source.
    """
    spbs_scores = [calculate_spbs(source, ignore_missing) for source in sources]
    
    if not spbs_scores:
        return 0.0
    return sum(spbs_scores) / len(spbs_scores)


def disinformation_rate(sources: List[str], ignore_missing=False) -> float:
    """
    (# Satire + # very low + # low) / (# all sources)
    """
    factuality_data = load_factuality_data()
    
    if ignore_missing:
        sources = [source for source in sources if url_to_domain(source) in factuality_data]
    
    disinformation_count = sum(
        1 for source in sources
        if FACTUALITY_MAPPING.get(factuality_data.get(url_to_domain(source), 'NOT RATED').lower(), 0) < 0
    )
    
    return disinformation_count / len(sources) if sources else 0.0


def credible_rate(sources: List[str], ignore_missing=False) -> float:
    """
    (# very high + # high + # mostly factual) / (# all sources)
    """
    factuality_data = load_factuality_data()
    
    if ignore_missing:
        sources = [source for source in sources if url_to_domain(source) in factuality_data]
    
    credible_count = sum(
        1 for source in sources
        if FACTUALITY_MAPPING.get(factuality_data.get(url_to_domain(source), 'NOT RATED').lower(), 0) > 0
    )
    
    return credible_count / len(sources) if sources else 0.0


def calculate_retrieval_metrics(sources: List[str], ignore_missing=False) -> dict:
    """
    Calculate retrieval metrics for a list of sources.
    
    :param sources: List of lists of sources (URLs)
    :return: Dictionary with retrieval metrics
    :rtype: dict
    
    This function calculates the Source Factuality Score (SFS), Source Political Bias Score (SPBS),
    and disinformation rate for the provided sources.
    """
    return {
        'sfs': calculate_scs(sources, ignore_missing),
        'spbs': calculate_spbs(sources, ignore_missing),
        'disinformation_rate': disinformation_rate(sources, ignore_missing),
        'credible_rate': credible_rate(sources, ignore_missing),
    }
    

def calculate_retrieval_metrics_all(sources: List[List[str]], ignore_missing=False) -> dict:
    """
    Calculate retrieval metrics for a list of lists of sources.
    
    :param sources: List of lists of sources (URLs)
    :return: Dictionary with retrieval metrics
    :rtype: dict
    
    This function calculates the overall Source Factuality Score (SFS), Source Political Bias Score (SPBS),
    and disinformation rate for the provided sources.
    """
    return {
        'sfs': overall_credibility_score(sources, ignore_missing),
        'spbs': overall_political_bias_score(sources, ignore_missing),
        'disinformation_rate': disinformation_rate([url for sublist in sources for url in sublist], ignore_missing),
        'credible_rate': credible_rate([url for sublist in sources for url in sublist], ignore_missing)
    }

