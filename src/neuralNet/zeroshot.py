import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset

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
    a = np.array(l_vector[:3])
    b = np.array(l_vector[3:])

    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    scale = np.array([sum_a, sum_b])
    aF, bF = np.exp(scale) / np.sum(np.exp(scale))
    
    #this should scale down only the less dominated group
    a_scaled = a * (aF / max(aF, bF))
    b_scaled = b * (bF / max(aF, bF))
    
    return np.concatenate([a_scaled, b_scaled]).tolist()

def process_teacher_batch(scene_batch, processed_data):
    """
    Processes a character-specific batch through BART to generate 
    the behavioral ground truth (L).
    """
    # Extract only the text for the model
    texts = [item[0] for item in scene_batch]
    indices = [item[1] for item in scene_batch]

    BASE_BOOST = 0.05  
    
    dataset = Dataset.from_dict({"text": texts})
    results = classifier(KeyDataset(dataset, "text"), candidate_labels=LABELS, multi_label=True)

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