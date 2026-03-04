import json
from pathlib import Path
from collections import defaultdict
from transformers import pipeline
from processData.textPipeline import  doc_container, registry #remove these pref after teast once we wanna use storage
from processData.sceneGenerator import scene_batch_generator
from src.fileIO import load_doc_container, load_registry
from config import WINDOW_SIZE
import skweak

#TODO: DEBATE IF NEED, THIS IS ADD TO spacy pipeline, not post bart, 
# it require additional helpers to add bart and handle type mismatch before it can be pitch against bart for a slightly better labelling
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

def lexicon_label_func(doc_container):
    """
    Yields ALL categories associated with a token in a single pass.
    """
    for doc in doc_container:
        for token in doc:
            # Get the list of categories for this word (returns [] if not found)
            categories = word_to_cats.get(token.text.lower(), [])
            
            for category in categories:
                # skweak will record multiple labels for the same token index
                yield token.i, token.i+1, category

# Wrap the function for skweak
lexicon_lf = skweak.heuristics.FunctionAnnotator("lexicon_labeling", lexicon_label_func)

#Helper to get span token from line_id, the output is a tuple of token spacy and skweak can use
def get_span_token(context):
    doc_id, local_sent_id = context
    currdoc = doc_container[doc_id]
    sentences = list(currdoc.sents)
   
    if local_sent_id > 1:
        first_sent = sentences[local_sent_id - 2]
    else:
        first_sent = sentences[0]
    if local_sent_id == len(sentences) -1:
        last_sent = sentences[local_sent_id]
    else:
        last_sent = sentences[local_sent_id +1]

    return first_sent.start, last_sent.end