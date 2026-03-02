import json
from pathlib import Path
from empath import Empath

lexicon = Empath()

CATEGORIES = {
    "logic": ["because", "therefore", "reason", "conclusion", "logic", "deduce"],
    "perception": ["observe", "notice", "see", "hear", "detect", "scent"],
    "knowledge": ["know", "fact", "evidence", "aware", "certainty", "truth"],
    "fear": ["afraid", "horror", "dread", "terror", "fright", "scared"],
    "desire": ["want", "hope", "ambition", "wish", "longing", "yearn"],
    "stress": ["pressure", "tense", "burden", "strained", "grief", "anxious"]
}
expanded_6d_lexicon = {}

for category_name, seed_words in CATEGORIES.items():
    # This line runs the NN for each category .create_category() (Word2vec)
    expanded_words = lexicon.create_category(category_name, seed_words, model="fiction")
    expanded_6d_lexicon[category_name] = expanded_words

storage_path = Path('data/6d_lexicon.json')
with open(storage_path, 'w') as f:
    json.dump(expanded_6d_lexicon, f, indent=4)

