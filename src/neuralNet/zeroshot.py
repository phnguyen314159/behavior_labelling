import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
from transformers import pipeline

from src.processData.sceneGenerator import scene_batch_gen
#from src.fileIO import load_doc_container, load_registry
from src.config import WINDOW_SIZE, BATCH_SIZE, LABELS

# --- PATH SETUP ---
# Locating the baked lexicon in the root 'data' folder
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent.parent
lexicon_path = root_dir / "data" / "6d_lexicon.json"

with open(lexicon_path, "r", encoding="utf-8") as f:
    behavior_lexicon = json.load(f)

word_to_cats = defaultdict(list)
for cat, words in behavior_lexicon.items():
    for word in words:
        word_to_cats[word.lower()].append(cat)

# Initialize the 'Teacher' model
# bart-large-mnli is heavy; batching is essential to stay efficient
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=0,
                      batch_size=BATCH_SIZE) # Uses GPU

#helper for weighted labels
def calculate_w_vector(l_vector):
    # 1. Split the 6D vector into Groups A and B
    group_a_raw = np.array(l_vector[:3])
    group_b_raw = np.array(l_vector[3:])
    
    # 2. Competitive Rank between A and B (Global Softmax)
    # This gives you aF and bF which sum to 1.0
    sum_a = np.sum(group_a_raw)
    sum_b = np.sum(group_b_raw)
    
    # Use softmax to make them competitive
    global_logits = np.array([sum_a, sum_b])
    aF, bF = np.exp(global_logits) / np.sum(np.exp(global_logits))
    
    # this should scale down only the less dominated group
    a_scaled = group_a_raw * (aF / max(aF, bF))
    b_scaled = group_b_raw * (bF / max(aF, bF))
    
    # 4. Re-assemble into final 6D w_vector
    return np.concatenate([a_scaled, b_scaled]).tolist()

def process_teacher_batch(scene_batch, processed_data):
    """
    Processes a character-specific batch through BART to generate 
    the behavioral ground truth (L).
    """
    # Extract only the text for the model
    texts = [item[0] for item in scene_batch]
    indices = [item[1] for item in scene_batch]

    # Configuration for the bias
    BASE_BOOST = 0.05  # The 'nudge' per word match
    
    # Multi-label inference: allows the model to score all 6 dimensions independently
    results = classifier(texts, candidate_labels=LABELS, multi_label=True)
    
    # Structure the output for the Embedding Store
    for i, res in enumerate(results):
        score_map = dict(zip(res['labels'], res['scores']))
        current_text = texts[i].lower()
        nudges = {label: 0.0 for label in LABELS}
        for word, cats in word_to_cats.items():
            if word in current_text:
                count = current_text.count(word)
                for cat in cats:
                    # Logarithmic scaling so 10 matches don't break the scale
                    nudges[cat] += BASE_BOOST * math.log1p(count)
        # Create the 6D vector from the scores
        # scores are returned in the order of LABELS
        # Ensure the vector matches your LABELS order exactly
        l_vector = []
        for lbl in LABELS:
            bart_score = score_map[lbl]
            
            # Final Score: BART probability + Lexicon Bias 
            final_score = bart_score + (nudges[lbl] * (1.0 - bart_score))
            l_vector.append(final_score)
        w_vector = calculate_w_vector(l_vector)
        
        processed_data.append({ 
            "context": indices[i], #context[0] is doc_id, context[1] is local sent_id
            "label_vector": l_vector,
            "weighted_vector": w_vector
        })
        
#return: update a container of labels from bart! items: character_name (also relative by book), list of tuple: (sent_id, labels vector)