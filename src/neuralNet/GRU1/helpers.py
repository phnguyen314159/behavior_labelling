import torch
import math
from pathlib import Path
from src.config import STEP

BASE_DIR = Path(__file__).resolve().parent

#all_results.get("ZSHOT", {}) -> sbert
#all_results.get("ENCODE", {}) -> bart

#we currently use a naive decay, we know the relative sent_id of each encoding in our list (and they are all pivoted on a char)
#so, we purely note down the distance between 2 neighboring encoding, if they close, low decay, if they far, high decay.
#ex: sent1, sent2, sent 3, sent 56, sent 57, sent 80 -> big decay for sent 56 encoding and sent 80 encoding
def prepare_and_save_chunks(all_res_list, chunk_size=100):
    """
    bart: { 'Name': [{'label_vector': [...], ...}, ...], ... }
    sbert: { 'Name': [ [768_dims], [768_dims], ... ], ... }
    """
    for e, all_res in enumerate(all_res_list):
        total_chunks = 0
        char_names = list(all_res.get("ENCODE", {}).keys())
        for char_name in char_names:
            n = len(all_res["ENCODE"][char_name])
            if n >= chunk_size:
                # math.ceil ensures we get the final partial chunk too
                total_chunks += math.ceil(n / chunk_size)
        all_encode = torch.empty((total_chunks, chunk_size, 768))
        all_lbls = torch.empty((total_chunks, chunk_size, 6))
        all_deltas = torch.empty((total_chunks, chunk_size))

        idx = 0
        for char_name in char_names:
            encodes = [item['obs_vector'] for item in all_res.get("ENCODE", {})[char_name]]
            lbls = [item['weighted_vector'] for item in all_res.get("ZSHOT", {})[char_name]]
            contexts = [item['context'] for item in all_res.get("ZSHOT", {})[char_name]]
            positions = [(c[0]*STEP) + c[1] for c in contexts]

            n = len(encodes)
            
            if n < chunk_size:
                continue
           
            for i in range(0, n, chunk_size):
                #check for the last chunk
                if i + chunk_size > n:
                    #if overshoot, take the last 'chunk_size' elements
                    start = n - chunk_size
                    end = n
                else:
                    start = i
                    end = i + chunk_size

                buffer = positions[start]
                buffer_positions = positions[start:end]
                all_deltas[idx] = torch.tensor([float(bp - buffer) for bp in buffer_positions])
                
                all_encode[idx] = torch.stack(encodes[start:end]) #since this is alr a list of tensors
                all_lbls[idx] = torch.tensor(lbls[start:end])
                idx += 1

        torch.save({'encodings': all_encode, 'labels': all_lbls, 'deltas': all_deltas}, BASE_DIR / f"distill_data{e}.pt")

def load_first(filename="distill_data0.pt"): #default to first book
    load_path = BASE_DIR / filename
    data = torch.load(load_path, weights_only=True)
    return data['encodings'], data['labels'], data['deltas']

def load_all(pattern: str = "*.pt"):
    load_path = BASE_DIR
    if not load_path.exists():
        raise FileNotFoundError(f"Folder not found: {load_path.resolve()}")

    for fp in sorted(load_path.glob(pattern)):
        data = torch.load(fp, weights_only=True)
        yield data['encodings'], data['labels'], data['deltas']