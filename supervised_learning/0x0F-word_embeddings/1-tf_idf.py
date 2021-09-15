#!/usr/bin/env python3
"""
Creates a TF-
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Module that returns features
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    X = vectorizer.fit_transform(sentences)

    features = vectorizer.get_feature_names()

    embeddings = X.toarray()

    return embeddings, features
