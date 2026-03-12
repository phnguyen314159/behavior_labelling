from src.neuralNet.zeroshot import process_teacher_batch
from src.processData.sceneGenerator import scene_batch_gen


def run_behavioral_pipeline(doc_container, registry):
    """
    Main loop to process all characters in the registry.
    Saves results into a dictionary keyed by character name.
    """
    # This will hold: { "CharacterName": [ {result_dict}, ... ], ... }
    all_bart_results = {name: [] for name in registry.keys()}

    # 2. Call the generator for this specific character
    # str, list
    batch_gen = scene_batch_gen(doc_container, registry)
    for target_name, scene_batch in batch_gen:
        # Process the batch and update the specific character's result list
        process_teacher_batch(scene_batch, all_bart_results[target_name])
        
    return all_bart_results