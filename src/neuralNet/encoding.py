import torch
from sentence_transformers import SentenceTransformer
from config import BATCH_SIZE

#TODO: SAROSH CHECK IF RUNNABLE AND MODIFY
# 1. Initialize SBERT
# 'all-mpnet-base-v2' is the gold standard for 768d embeddings.
# It is significantly faster than BART but still benefits from batching.
encoder = SentenceTransformer('all-mpnet-base-v2', device='cuda')

#TODO: ADD A PUBLIC CONTAINER AS PREFERENCE SO INSTEAD OF RETURN A BATCH OF RESULT, WE UPDATE THAT CONTAINER
def process_observation_batch(character_name, scene_batch):
    """
    Processes a character-specific batch through SBERT to generate 
    the 768d observation vectors (obs_t).
    """
    # Extract text and indices
    texts = [item[0] for item in scene_batch]
    indices = [item[1] for item in scene_batch]
    
    # 2. Batch Encoding
    # convert_to_tensor=True gives you the 16x768 matrix immediately.
    # show_progress_bar=False keeps your logs clean during the 'busy work'.
    embeddings = encoder.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        convert_to_tensor=False, 
        show_progress_bar=False
    )

    processed_observations = []
    for i in range(len(embeddings)):
        processed_observations.append({
            "sent_idx": indices[i],
            "obs_vector": embeddings[i] # This is a 768d tensor
        })
        
    return processed_observations
