import torch
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#all_results.get("ZSHOT", {}) -> sbert
#all_results.get("ENCODE", {}) -> bart
def prepare_and_save_chunks(all_res, bart_data_dict, sbert_data_dict, filename="distill_data.pt", chunk_size=100):
    """
    bart_data_dict: { 'Name': [{'label_vector': [...], ...}, ...], ... }
    sbert_data_dict: { 'Name': [ [768_dims], [768_dims], ... ], ... }
    """

    total_chunks = 0
    for char_name in sbert_data_dict:
        n = len(sbert_data_dict[char_name])
        if n >= chunk_size:
            # math.ceil ensures we get the final partial chunk too
            total_chunks += math.ceil(n / chunk_size)
    all_encode = torch.empty((total_chunks, chunk_size, 768))
    all_lbls = torch.empty((total_chunks, chunk_size, 6))

    idx = 0
    for char_name in sbert_data_dict:
        encodes = sbert_data_dict[char_name]
        lbls = [item['weighted_vector'] for item in bart_data_dict[char_name]]
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
            
            # Fill pre-allocated memory directly (No torch.stack needed!)
            sbert_tensor[idx] = torch.tensor(encodes[start:end])
            bart_tensor[idx] = torch.tensor(lbls[start:end])
            idx += 1

    torch.save({'encodings': sbert_tensor, 'labels': bart_tensor}, BASE_DIR / filename)

def load_distill_data(filename="distill_data.pt"):
    load_path = BASE_DIR / filename
    data = torch.load(load_path, weights_only=True)
    return data['encodings'], data['labels']