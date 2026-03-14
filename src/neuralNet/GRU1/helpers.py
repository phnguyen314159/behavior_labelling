import torch
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#all_results.get("ZSHOT", {}) -> sbert
#all_results.get("ENCODE", {}) -> bart
def prepare_and_save_chunks(all_res, filename="distill_data.pt", chunk_size=100):
    """
    bart: { 'Name': [{'label_vector': [...], ...}, ...], ... }
    sbert: { 'Name': [ [768_dims], [768_dims], ... ], ... }
    """

    total_chunks = 0
    for char_name in all_res:
        n = len(all_res[char_name])
        if n >= chunk_size:
            # math.ceil ensures we get the final partial chunk too
            total_chunks += math.ceil(n / chunk_size)
    all_encode = torch.empty((total_chunks, chunk_size, 768))
    all_lbls = torch.empty((total_chunks, chunk_size, 6))

    idx = 0
    char_names = list(all_res.get("ENCODE", {}).keys())
    for char_name in char_names:
        encodes = [item['obs_vector'] for item in all_res.get("ENCODE", {})[char_name]]
        lbls = [item['weighted_vector'] for item in all_res.get("ZSHOT", {})[char_name]]
        n = len(encodes)
        
        if n < chunk_size:
            continue
            
        # Standard sliding window over range
        for i in range(0, n, chunk_size):
            # Check for the last chunk
            if i + chunk_size > n:
                # If we'd overshoot, take the last 'chunk_size' elements
                start = n - chunk_size
                end = n
            else:
                start = i
                end = i + chunk_size
            all_encode[idx] = torch.stack(encodes[start:end]) #since this is alr a list of tensors
            all_lbls[idx] = torch.tensor(lbls[start:end])
            idx += 1

    torch.save({'encodings': all_encode, 'labels': all_lbls}, BASE_DIR / filename)

def load_distill_data(filename="distill_data.pt"):
    load_path = BASE_DIR / filename
    data = torch.load(load_path, weights_only=True)
    return data['encodings'], data['labels']