import pandas as pd
import numpy as np
from collections import defaultdict

def cosine_score(query: list, result: pd.DataFrame, dictionary_index: dict, tfidf_index: defaultdict, top_k: int = 10) -> pd.DataFrame:
    scores = defaultdict(float)
    lengths = defaultdict(float)
    query_term_ids = [key for key, value in dictionary_index.items() if value in query]
    for term_id in query_term_ids:
        if term_id in tfidf_index:
            w_t_q = 1.0
            postings = tfidf_index[term_id]
            for doc_id, tfidf in postings:
                scores[doc_id] += tfidf * w_t_q
    for term_id, postings in tfidf_index.items():
        for doc_id, tfidf in postings:
            lengths[doc_id] += tfidf ** 2
    for doc_id in scores.keys():
        if lengths[doc_id] > 0:
            scores[doc_id] /= np.sqrt(lengths[doc_id])
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_results = result[result['id'].isin([doc_id for doc_id, _ in top_docs])]
    top_results['cosine_score'] = pd.Series(top_results['id'].map(dict(top_docs)))
    return top_results.sort_values(by='cosine_score', ascending=False)

# ====================================================================================================

# COSINESCORE(q)
#   float Scores[N] = 0
#   Initialize Length[N]
#   for each query term t
#   do calculate w(t,q) and fetch postings list for t
#     for each pair(d, tf(t,d)) in postings list
#     do Scores[d] += wft,d × wt,q
#   Read the array Length[d]
#   for each d
#   do Scores[d] = Scores[d]/Length[d]
#   return Top K components of Scores[]