def generate_kgrams(term: str, k: int = 2) -> list:
    term = f"${term}$"
    return [term[i:i+k] for i in range(len(term) - k + 1)]

def jaccard_coefficient(query_kgrams: list, term_kgrams: list) -> int:
    intersection = len(set(query_kgrams) & set(term_kgrams))
    union = len(set(query_kgrams) | set(term_kgrams))
    return intersection / union if union > 0 else 0
