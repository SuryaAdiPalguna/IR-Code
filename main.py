import pandas as pd
import pickle
import gzip
import streamlit as st
from collections import defaultdict
from modul.preprocessing import case_folding, tokenizing, stopword_removal, normalization, stemming
from modul.scoring import cosine_score
from modul.spelling_correction import generate_kgrams, jaccard_coefficient

def preprocessing(query: str) -> list:
  query = case_folding(query)
  query = tokenizing(query)
  query = stopword_removal(query)
  query = normalization(query)
  query = stemming(query)
  return query

def find_document(query: list) -> pd.DataFrame:
  with gzip.open(filename='index/metadata_index.pkl.gz', mode='rb') as f:
    df: pd.DataFrame = pickle.load(f)
  with gzip.open(filename='index/inverted_index.pkl.gz', mode='rb') as f:
    inverted_index: defaultdict = pickle.load(f)
  relevant_ids = set()
  for term in query:
    if term in inverted_index:
      relevant_ids.update(inverted_index[term])
  result_df = df[df['id'].isin(relevant_ids)]
  return result_df

def spelling_correction(query: list) -> list:
  with gzip.open(filename='index/dictionary_index.pkl.gz', mode='rb') as f:
    dictionary_index: dict = pickle.load(f)
  with gzip.open(filename='index/kgram_index.pkl.gz', mode='rb') as f:
    kgram_index: dict = pickle.load(f)
  corrected_query = []
  for word in query:
    query_kgrams = generate_kgrams(word)
    candidate_term_ids = set()
    for kgram in query_kgrams:
      if kgram in kgram_index:
        candidate_term_ids.update(kgram_index[kgram])
    candidate_term_dict = {key: dictionary_index[key] for key in candidate_term_ids if key in dictionary_index}
    best_match: str
    best_score = 0
    candidate_term = list(candidate_term_dict.values())
    for term in candidate_term:
      term_kgrams = generate_kgrams(term)
      score = jaccard_coefficient(query_kgrams, term_kgrams)
      if score > best_score:
        best_match = term
        best_score = score
    corrected_query.append(best_match if best_match else word)
  return corrected_query

def scoring_document(query: list, result: pd.DataFrame, top_k: int = 10):
  with gzip.open(filename='index/dictionary_index.pkl.gz', mode='rb') as f:
    dictionary_index: dict = pickle.load(f)
  with gzip.open(filename='index/tfidf_index.pkl.gz', mode='rb') as f:
    tfidf_index: defaultdict = pickle.load(f)
  return cosine_score(query, result, dictionary_index, tfidf_index, top_k)

st.set_page_config(page_title="MESIN PENCARI BERITA", layout="centered")
st.title("MESIN PENCARI BERITA")
query = st.text_input("Masukkan kata pencarian:")

if query:
  query_processing = preprocessing(query)
  correct_query = spelling_correction(query_processing)

  if query_processing == correct_query:
    result = find_document(query_processing)
    result = scoring_document(query_processing, result, top_k=10)
  else:
    st.write(f"Mungkin yang dimaksud: **{correct_query}**")
    result = find_document(correct_query)
    result = scoring_document(correct_query, result, top_k=10)

  st.write("## Hasil Pencarian:")
  if len(result) > 0:
    for index, row in result.iterrows():
      st.write(f'[{row['url'].split('://')[1].split('/')[0]}]({row['url']}) > {row['source']}')
      st.write(f"#### [{row['title']}]({row['url']})")
      st.write(f"{row['content'][:200]}...")
      st.write(f"**Relevansi:** {row['cosine_score']:.4f}")
      st.write("---")
  else:
    st.write("Tidak ada dokumen yang relevan ditemukan.")
