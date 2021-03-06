#!/usr/bin/env python3
"""
Creates and trains a gensim fastText model
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Returns: gensim fastText trained model
    """
    model = FastText(sentences=sentences, min_count=min_count,
                     iter=iterations, size=size,
                     window=window, sg=cbow,
                     seed=seed, negative=negative)

    model.train(sentences=sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
