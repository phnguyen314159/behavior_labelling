import spacy
import json
from spacy.tokens import DocBin
from pathlib import Path

def save_doc_container(doc_container, filepath):
    """
    Serializes a list of (Doc, context) tuples. 
    Expects a dynamic filepath like f"data/processed_docs/{counter}.spacy"
    """
    # Initialize DocBin with store_user_data=True to preserve the context dictionary
    doc_bin = DocBin(store_user_data=True)

    for doc, context in doc_container:
        # Embed the 'context' dictionary (offset, sent_range, etc.) into the Doc's user_data
        # This ensures metadata stays attached to the specific Doc during serialization
        doc.user_data["context"] = context
        doc_bin.add(doc)

    # Ensure the target directory exists, make if not
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc_bin.to_disk(save_path)


def load_doc_container(nlp, filepath):
    """
    Loads a DocBin from disk and reconstructs the (Doc, context) list format.
    """
    load_path = Path(filepath)
    if not load_path.exists():
        raise FileNotFoundError(f"No serialized DocBin found at: {filepath}")

    # Load the binary data from disk
    doc_bin = DocBin().from_disk(load_path)
    
    # nlp.vocab is required to reconstruct the Doc objects with correct string hashes
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    # Reconstruct the original (Doc, context) tuple structure
    reconstructed = []
    for doc in docs:
        # Retrieve the context dictionary we stored in user_data
        context = doc.user_data.get("context", {})
        reconstructed.append((doc, context))

    return reconstructed

#registry
def save_registry(registry, filepath):
    """
    Saves the final registry dictionary to a JSON file.
    """
    save_path = Path(filepath)
    # Ensure the directory exists before writing
    save_path.parent.mkdir(parents=True, exist_ok=True) 

    with open(save_path, "w", encoding="utf-8") as f:
        # indent=4 makes it readable for debugging
        json.dump(registry, f, indent=4, ensure_ascii=False)
    
    print(f"Registry successfully saved to {save_path}")

def load_registry(filepath):
    """
    Loads the registry dictionary from a JSON file.
    """
    load_path = Path(filepath)
    if not load_path.exists():
        print(f"No registry found at {filepath}, returning empty dict.")
        return {}

    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)