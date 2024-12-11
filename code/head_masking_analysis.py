import pandas as pd
import numpy as np
from collections import defaultdict

def head_masking_analysis(filename, output_path):
    
    head_masked_probs = pd.read_csv(filename)
    head_masked_probs = head_masked_probs.sort_values(by = ["Text", "mask"], axis = 0)
    head_masked_probs = head_masked_probs.drop_duplicates(subset=["Text", "mask"])
    delta = {
    "LABEL_0": defaultdict(list),
    "LABEL_1": defaultdict(list),
    "LABEL_2": defaultdict(list),
    "LABEL_3": defaultdict(list),
    "LABEL_4": defaultdict(list),
    "LABEL_5": defaultdict(list)
}
    sentence_orig_prob = {}

    for _, row in head_masked_probs.iterrows():
        idx = row["True"][-1]
        if row["mask"] == -1:
            sentence_orig_prob[row["Text"]] = row[idx]
        else:
            diff = row[idx] - sentence_orig_prob[row["Text"]]
            delta[row["True"]][row["mask"]].append(diff)

    for label in delta:
        for head in delta[label]:
            delta[label][head] = np.mean(delta[label][head])

    np.save(output_path, delta) 