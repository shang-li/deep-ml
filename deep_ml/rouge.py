from collections import Counter

def rouge_1(reference: str, candidate: str) -> (float, float, float):
    ref_list = reference.split(" ")
    cand_list = candidate.split(" ")
    ref_dict = Counter(ref_list)
    cand_dict = Counter(cand_list)

    noverlaps_ref = 0  # recall = noverlaps / ref_n_words
    noverlaps_cand = 0 # precision = noverlaps_cand / cand_n_words

    for word, cnt in ref_dict.items():
        noverlaps_ref += min(cand_dict.get(word, 0), cnt)
    for word, cnt in cand_dict.items():
        noverlaps_cand += min(ref_dict.get(word, 0), cnt)
    
    recall = noverlaps_ref / len(ref_list)
    precision = noverlaps_cand / len(cand_list)
    f1 = 2 / (1/recall + 1/precision)
    return recall, precision, f1

print(rouge_1("the cat sat on the mat", "the cat is on the mat"))