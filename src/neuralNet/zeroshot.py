import torch
from transformers import pipeline
from processData.sceneGenerator import bart_prep_generator, doc_container, registry #remove these pref after teast once we wanna use storage
from src.fileIO import load_doc_container, load_registry

#TODO: WIP, NEED TO CHECK HARDWARE USAGE + MAKE SENSE OF THE PIPELINE + CLARIFY OUTPUT -> PLAN TO PLUG OUTPUT IMMEDIATELY INTO SKWEAK
def run_zeroshot_labelling(target_character):
    # 1. Setup candidate labels from your behavior vector
    candidate_labels = ["Logic", "Perception", "Knowledge", "Fear", "Desire", "Stress"]
    
    # 2. Initialize the BART zero-shot pipeline
    # Using 'facebook/bart-large-mnli' as it is the standard for zero-shot classification
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", 
                          model="facebook/bart-large-mnli", 
                          device=device)

    # 3. Load processed data
    """
    from src.processData.textPipeline import nlp
        #place holder for proper loop of getting from storage, for test we can just do from mem
    numDoc, numReg = 0
    doc_container = load_doc_container(nlp, f"data/processed_docs/{numDoc}.spacy")
    registry = load_registry(f"data/processed_docs/{numReg}.json")
    """

    # 4. Processing Loop
    results = []
    # Use the generator to get the context-aware prompts
    for prompt in bart_prep_generator(doc_container, registry, target_character):
        # We classify the prompt based on your 6 dimensions
        prediction = classifier(prompt, candidate_labels, multi_label=True)
        
        # Organize the output to match your 6D vector format: L = [l, p, k, f, d, s]
        #
        scores = {label: score for label, score in zip(prediction['labels'], prediction['scores'])}
        vector_6d = [
            scores.get("Logic", 0),
            scores.get("Perception", 0),
            scores.get("Knowledge", 0),
            scores.get("Fear", 0),
            scores.get("Desire", 0),
            scores.get("Stress", 0)
        ]
        
        results.append({
            "prompt": prompt,
            "vector": vector_6d,
            "top_label": prediction['labels'][0]
        })
        
        print(f"Character: {target_character} | Vector: {vector_6d}")

    return results