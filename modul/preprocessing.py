import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from mpstemmer import MPStemmer

def cleaning_data(df: pd.DataFrame) -> pd.DataFrame:
  df = df.dropna()
  df = df.drop_duplicates()
  return df

def case_folding(text: str) -> str:
  return text.lower()

def tokenizing(text: str) -> list:
  return text.split()

def stopword_removal(text: list) -> list:
  nltk.download('stopwords')
  stop_words = set(stopwords.words('indonesian'))
  return [word for word in text if word not in stop_words]

def normalization(text: list) -> list:
  text = ' '.join(text)
  # text = re.sub(r'^.*- ', ' ', text)
  # remove_hashtags_mentions_urls
  text = re.sub(r'#\w+', ' ', text)
  text = re.sub(r'@\w+', ' ', text)
  text = re.sub(r'http\S+', ' ', text)
  text = re.sub(r'www\S+', ' ', text)
  # remove_punctuation
  text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
  return text.split()

def stemming(text: list) -> list:
  stemmer = MPStemmer()
  return [stemmer.stem_kalimat(word) for word in text]
