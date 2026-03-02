import torch
from transformers import pipeline
from processData.textPipeline import  doc_container, registry #remove these pref after teast once we wanna use storage
from processData.sceneGenerator import scene_batch_generator
from src.fileIO import load_doc_container, load_registry
from config import BATCH_SIZE, LABELS

# Initialize the 'Teacher' model
# bart-large-mnli is heavy; batching is essential to stay efficient
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=0,
                      batch_size=BATCH_SIZE) # Uses GPU


#TODO: ADD A PUBLIC CONTAINER AS PREFERENCE SO INSTEAD OF RETURN A BATCH OF RESULT, WE UPDATE THAT CONTAINER
def process_teacher_batch(character_name, scene_batch):
    """
    Processes a character-specific batch through BART to generate 
    the behavioral ground truth (L).
    """
    # Extract only the text for the model
    texts = [item[0] for item in scene_batch]
    indices = [item[1] for item in scene_batch]
    
    # Multi-label inference: allows the model to score all 6 dimensions independently
    results = classifier(texts, candidate_labels=LABELS, multi_label=True)
    
    # Structure the output for the Embedding Store
    processed_data = []
    for i, res in enumerate(results):
        # Create the 6D vector from the scores
        # scores are returned in the order of LABELS
        l_vector = res['scores'] 
        
        processed_data.append({
            "sent_idx": indices[i],
            "label_vector": l_vector
        })
        
    return processed_data