from sentence_transformers import SentenceTransformer
from src.config import BATCH_SIZE

#TODO: SAROSH CHECK IF RUNNABLE AND MODIFY
# 1. Initialize SBERT
# 'all-mpnet-base-v2' is the gold standard for 768d embeddings.
# It is significantly faster than BART but still benefits from batching.
encoder = SentenceTransformer('all-mpnet-base-v2', device='cuda')

def process_observation_batch(scene_batch, processed_data):
    """
    Processes a character-specific batch through SBERT to generate 
    the 768d observation vectors (obs_t).
    """
    # Extract text and indices
    texts = [item[0] for item in scene_batch] #text of the scene
    indices = [item[1] for item in scene_batch] #(doc_id, local_s_idx)
    

    embeddings = encoder.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        convert_to_tensor=True, #fix to avoid heavy convert to cpu ram
        show_progress_bar=False
    )

    for i in range(len(embeddings)):
        processed_data.append({
            "context": indices[i],
            "obs_vector": embeddings[i] # This is a 768d tensor
        })
