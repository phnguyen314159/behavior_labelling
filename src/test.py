import random
import sys
import os
import pytest
from src.processData.textPipeline import iter_books, book_process, global_ent, cluster_container, process_registry
from src.NNrun import run_behavioral_pipeline
from src.neuralNet.zeroshot import calculate_w_vector

def test_pipeline_with_real_data():
    """
    Test the pipeline using the actual files in data/book/test.
    """
    # 1. Clear global containers to start fresh for the test
    global_ent.clear()
    cluster_container.clear() 
    
    # 2. Get the first book from the test folder
    # iter_books yields (book_id, text)
    books = list(iter_books(mode="test"))
    
    # Ensure there is at least one file in data/book/test/
    assert len(books) > 0, "No .clean.txt files found in data/book/test"
    
    book_id, text = books[0]
    print(f"\nTesting with book: {book_id}")

    # 3. Process the book
    # This runs sliding_window, NER extraction, and coref logic
    doc_container = book_process(text)

    # 4. Verify Results
    # Check that we actually extracted data
    assert len(doc_container) > 0, "Pipeline failed to produce doc chunks"
    assert len(global_ent) > 0, "No PERSON entities were extracted from the test file"

    # 5. Schema Validation for the first entity
    sample_ent = global_ent[0]
    required_keys = [
        "type", "text", "global_start", "global_end", 
        "doc_id", "doc_token_pos", "sentence_id"
    ]
    
    for key in required_keys:
        assert key in sample_ent, f"Key '{key}' missing from extracted entity"

    # Verify doc_token_pos is a tuple for span access
    assert isinstance(sample_ent["doc_token_pos"], tuple)
    
    print(f"Test Passed: Extracted {len(global_ent)} entities from {book_id}")

    # 6. Test Registry Builder
    registry = process_registry(global_ent, cluster_container)
    
    # Verify the registry output schema
    assert isinstance(registry, dict), "Registry must be a dictionary"
    
    if len(global_ent) > 0:
        assert len(registry) > 0, "Registry failed to build unique person buckets"
        
        # Check schema of the first unique person bucket
        first_person = list(registry.keys())[1]
        assert "references" in registry[first_person], "Registry entry missing 'references' list"
        
        # Check schema of a reference item
        if len(registry[first_person]["references"]) > 0:
            sample_ref = registry[first_person]["references"][0]
            # We removed the "type" assertion here!
            assert "text" in sample_ref, "Reference missing 'text' key"
            assert "global_char_pos" in sample_ref, "Reference missing 'global_char_pos' key"
            
            
        print(f"Test Passed: Built registry with {len(registry)} unique person(s).")

        # --- Print out the first registry entry to the terminal ---
        result = f"total of {len(registry[first_person]["references"])} occurances of {first_person}"

        print(result)

# --- NEW IMPORTS ---
def test_zeroshot_behavioral_pipeline():
    """
    Test the full integration from text processing to behavioral vector generation.
    """
    # 1. Setup Data (Clear previous state)
    global_ent.clear()
    cluster_container.clear()
    
    books = list(iter_books(mode="test"))
    book_id, text = books[0]
    
    # Process a limited amount of text for a faster test
    doc_container = book_process(text[:20000]) 
    registry = process_registry(global_ent, cluster_container)

    # 2. Run Behavioral Pipeline
    # This invokes NNrun -> sceneGenerator -> zeroshot
    all_results = run_behavioral_pipeline(doc_container, registry)
    
    # 3. Assertions
    assert isinstance(all_results, dict), "Results should be a dictionary keyed by character name"
    
    if len(registry) > 0:
        first_char = list(registry.keys())[0]
        assert first_char in all_results, f"No results found for {first_char}"
        
        if len(all_results[first_char]) > 0:
            sample = all_results[first_char][0]
            assert "label_vector" in sample, "Missing label_vector in output"
            assert "weighted_vector" in sample, "Missing weighted_vector in output"
            assert len(sample["label_vector"]) == 6, "Vector length must match 6D labels"
    
    candidate = list(all_results.keys())[:2]

    print(f"{'Character':<12} | {'Label Vector (First 3)':<30} | {'Weighted Vector (First 3)'}")
    print("-" * 85)

    for char in candidate:  # Slicing to get only the first 3 dict entries
        data = all_results[char]
        for entry in data[:3]:
            l_vec_str = str([round(float(x), 4) for x in entry['label_vector']])
            w_vec_str = str([round(float(x), 4) for x in entry['weighted_vector']])
        
            print(f"{char:<12} | {l_vec_str:<55} | {w_vec_str}")
        print("-" * 85)

def test_vector_math_logic():
    """Verify the competitive softmax logic for Group A and B."""
    # Group A strong, Group B weak
    input_v = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1]
    weighted = calculate_w_vector(input_v)
    
    # Group A should remain dominant
    assert sum(weighted[:3]) > sum(weighted[3:])
    assert len(weighted) == 6

if __name__ == "__main__":
    #run from code as root
    #python -m src.test
    #if you are testing bart, use -b, if for pipe (heavy) use -p
    import argparse
    parser = argparse.ArgumentParser() # No description needed
    parser.add_argument("-p", "--pipe", action="store_true")
    parser.add_argument("-b", "--bart", action="store_true")
    args = parser.parse_args()

    if args.bart:
        test_zeroshot_behavioral_pipeline()
        test_vector_math_logic()
    if args.pipe:
        test_pipeline_with_real_data()
    