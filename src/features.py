import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(items: pd.DataFrame, max_features: int = 5000, ngram_range=(1,2)):
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english"
    )
    X = vec.fit_transform(items["text"].fillna(""))
    return vec, X