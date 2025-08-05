from deep_ml import compute_tf_idf
import numpy as np
import math

def test_td_idf():
    corpus = [["at", "home"], ["at", "school"], ["at", "office"]]
    query = ["at", "home", "kitchen"]
    out = compute_tf_idf(corpus, query)
    expected_out = [[0.5, 0.5 * (math.log(4/2) + 1), 0], [0.5, 0, 0], [0.5, 0, 0]]
    np.testing.assert_allclose(out, expected_out)