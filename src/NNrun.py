from neuralNet.zeroshot import process_teacher_batch
from processData.sceneGenerator import scene_batch_gen


def run_behavioral_pipeline(doc_container, registry):
    """
    Main loop to process all characters in the registry.
    Saves results into a dictionary keyed by character name.
    """
    # This will hold: { "CharacterName": [ {result_dict}, ... ], ... }
    all_bart_results = {}

    for target_name in registry:
        
        # 1. Initialize the specific list for this character
        character_results = []
        
        # 2. Call the generator for this specific character
        # Assumes scene_batch_generator takes (doc_container, character_name)
        batch_gen = scene_batch_gen(doc_container, target_name)
        
        for scene_batch in batch_gen:
            # 3. Process the batch and update character_results in-place
            process_teacher_batch(target_name, scene_batch, character_results)
            
        # 4. Store the complete character list in the main container
        all_bart_results[target_name] = character_results
        
    return all_bart_results