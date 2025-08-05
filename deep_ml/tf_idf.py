import numpy as np
from typing import List
from collections import Counter

def compute_tf_idf(corpus: List[List[str]], query: List[str]) -> List[List[float]]:
    docs_total = len(corpus)
    idf = [0] * len(query)
    tf = [[0] * len(query) for _ in range(docs_total)]

    for i, doc in enumerate(corpus):
        corpus_word_cnt = Counter(doc)
        corpus_words_total = len(doc)

        for j, word in enumerate(query):
            tf[i][j] = corpus_word_cnt.get(word, 0) / corpus_words_total
            idf[j] = idf[j] + 1 if tf[i][j] > 0 else idf[j]
    idf = np.array(idf).reshape(1, -1)
    tf = np.array(tf)
    #import pdb; pdb.set_trace()
    return tf * (np.log((docs_total + 1) / (idf + 1)) + 1)
    


