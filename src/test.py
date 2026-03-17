
import random
import sys
import os
import pytest
from src.processData.textPipeline import iter_books, book_process, global_ent, cluster_container, process_registry
from src.NNrun import data_pipeline_helper, models
from src.neuralNet.zeroshot import calculate_w_vector

def test_pipeline_with_real_data():
    """
    Test the pipeline using the actual files in data/book/test.
    """
    global_ent.clear()
    cluster_container.clear() 
    
    # iter_books yields (book_id, text)
    books = list(iter_books(mode="test"))
    
    # Ensure there is at least one file in data/book/test/
    assert len(books) > 0, "No .clean.txt files found in data/book/test"
    
    book_id, text = books[0]
    print(f"\nTesting with book: {book_id}")

    # This runs sliding_window, NER extraction, and coref logic
    doc_container = book_process(text)

    # Check that we actually extracted data
    assert len(doc_container) > 0, "Pipeline failed to produce doc chunks"
    assert len(global_ent) > 0, "No PERSON entities were extracted from the test file"

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

    registry = process_registry(global_ent, cluster_container)
    
    assert isinstance(registry, dict), "Registry must be a dictionary"
    
    if len(global_ent) > 0:
        assert len(registry) > 0, "Registry failed to build unique person buckets"
        
        first_person = list(registry.keys())[1]
        assert "references" in registry[first_person], "Registry entry missing 'references' list"
        
        # Check schema of a reference item
        if len(registry[first_person]["references"]) > 0:
            sample_ref = registry[first_person]["references"][0]
            assert "text" in sample_ref, "Reference missing 'text' key"
            assert "global_char_pos" in sample_ref, "Reference missing 'global_char_pos' key"
            
            
        print(f"Test Passed: Built registry with {len(registry)} unique person(s).")

        # --- Print out the first registry entry to the terminal ---
        result = f"total of {len(registry[first_person]["references"])} occurances of {first_person}"

        print(result)

def test_zeroshot_behavioral_pipeline():
    """
    Test the full integration from text processing to behavioral vector generation.
    """
    global_ent.clear()
    cluster_container.clear()
    
    books = list(iter_books(mode="test"))
    book_id, text = books[0]
    
    # Process a limited amount of text for a faster test
    doc_container = book_process(text[:20000]) 
    registry = process_registry(global_ent, cluster_container)

    all_results = data_pipeline_helper(doc_container, registry, [models.ZSHOT])
    zshot_results = all_results.get("ZSHOT", {})
    
    assert isinstance(all_results, dict), "Outer result should be a task dictionary"
    assert isinstance(zshot_results, dict), "ZSHOT results should be a char dictionary"
    
    if len(registry) > 0:
        first_char = list(registry.keys())[0]
        # Check inside the ZSHOT sub-dictionary
        assert first_char in zshot_results, f"No ZSHOT results found for {first_char}"
        
        if len(zshot_results[first_char]) > 0:
            sample = zshot_results[first_char][0]
            assert "label_vector" in sample, "Missing label_vector in output"
            assert "weighted_vector" in sample, "Missing weighted_vector in output"
            assert len(sample["label_vector"]) == 6, "Vector length must match 6D labels"

    candidate = list(zshot_results.keys())[:2]

    print(f"\n{'Character':<12} | {'Label Vector':<55} | {'Weighted'}")
    print("-" * 120)

    for char in candidate:
        data = zshot_results[char] # Access data from the sub-dict
        for entry in data[:2]:
            l_vec_str = str([round(float(x), 4) for x in entry['label_vector']])
            w_vec_str = str([round(float(x), 4) for x in entry['weighted_vector']])
            print(f"{char:<12} | {l_vec_str:<55} | {w_vec_str}")
        print("-" * 120)

    for char in candidate:  # Slicing to get only the first 3 dict entries
        data = zshot_results[char]
        for entry in data[:3]:
            l_vec_str = str([round(float(x), 4) for x in entry['label_vector']])
            w_vec_str = str([round(float(x), 4) for x in entry['weighted_vector']])
        
            print(f"{char:<12} | {l_vec_str:<55} | {w_vec_str}")
        print("-" * 85)

def test_encoding():
    global_ent.clear()
    cluster_container.clear()
    
    books = list(iter_books(mode="test"))
    book_id, text = books[0]

    doc_container = book_process(text[:20000]) 
    registry = process_registry(global_ent, cluster_container)

    all_results = data_pipeline_helper(doc_container, registry, [models.ENCODE])
    encode_results = all_results.get("ENCODE", {})
    
    if len(registry) > 0:
        first_char = list(registry.keys())[0]
        assert first_char in encode_results, f"No ENCODE results found for {first_char}"
        
        char_data = encode_results[first_char]
        if len(char_data) > 0:
            sample = char_data[0]
            
            # Key checks based on your process_observation_batch code
            assert "context" in sample, "Missing context (indices) in output"
            assert "obs_vector" in sample, "Missing obs_vector in output"
            
            # Check the dimensionality (SBERT is usually 768)
            vector_len = len(sample["obs_vector"])
            assert vector_len == 768, f"Expected 768d vector, got {vector_len}d"

    # Since 768 is too long to print, we show just the first 5 dimensions
    candidate = list(encode_results.keys())[:2]

    print(f"\n{'Character':<12} | {'Context (ID, Idx)':<20} | {'Obs Vector (First 5)'}")
    print("-" * 90)

    for char in candidate:
        data = encode_results[char]
        for entry in data[:3]:
            # Formatting the context and the vector slice
            ctx_str = str(entry['context'])
            v_slice = [round(float(x), 4) for x in entry['obs_vector'][:5]]
            v_str = f"{str(v_slice)[:-1]} ...]" # Add ellipsis to show it's truncated
            
            print(f"{char:<12} | {ctx_str:<20} | {v_str}")
        print("-" * 90)
        
def cache_zeroshot_pipeline():
    """
    OVERNIGHT RUN: Runs ONLY the heavy BART pipeline across multiple books and caches the results.
    """
    print("\n--- Starting Heavy ZSHOT Caching (Validation Set) ---")
    
    books = list(iter_books(mode="validation"))
    
    books_to_process = books[:3]
    print(f"Found {len(books)} validation books. Processing the first 3...")
    
    master_zshot_data = {}
    
    for i, (book_id, text) in enumerate(books_to_process):
        print(f"\n[{i+1}/3] Processing book: {book_id}")
        
        global_ent.clear()
        cluster_container.clear()
        
        doc_container = book_process(text) 
        registry = process_registry(global_ent, cluster_container)

        print(f"Running heavy BART zero-shot pipeline for {book_id}...")
        all_results = data_pipeline_helper(doc_container, registry, [models.ZSHOT])
        book_zshot_data = all_results.get("ZSHOT", {})
        
        for char_name, data_list in book_zshot_data.items():
            unique_char_name = f"{book_id}_{char_name}"
            
            if unique_char_name not in master_zshot_data:
                master_zshot_data[unique_char_name] = []
                
            master_zshot_data[unique_char_name].extend(data_list)
            
    print("\n--- All books processed! Saving Cache ---")
    from src.neuralNet.GRU1.helpers import save_zshot_cache
    save_zshot_cache(master_zshot_data, "zshot_cache_val_3books.pt")


def generate_distill_dataset():
    """
    FAST RUN: Loads the cached BART data, runs SBERT across the same 3 books, and stitches them together.
    """
    print("\n--- Starting Data Extraction for GRU (Validation Set) ---")
    
    from src.neuralNet.GRU1.helpers import load_zshot_cache, prepare_and_save_chunks
    
    try:
        print("Loading cached ZSHOT data...")
        bart_data_dict = load_zshot_cache("zshot_cache_val_3books.pt")
    except FileNotFoundError:
        print("Error: Cache not found! Run with -cz first to generate the ZSHOT cache.")
        return

    books = list(iter_books(mode="validation"))
    books_to_process = books[:3]
    
    master_encode_data = {}

    for i, (book_id, text) in enumerate(books_to_process):
        print(f"\n[{i+1}/3] Processing book for SBERT: {book_id}")
        
        global_ent.clear()
        cluster_container.clear()
        
        doc_container = book_process(text) 
        registry = process_registry(global_ent, cluster_container)

        print(f"Running fast SBERT encoding pipeline for {book_id}...")
        book_results = data_pipeline_helper(doc_container, registry, [models.ENCODE])
        book_encode_data = book_results.get("ENCODE", {})
        
        for char_name, data_list in book_encode_data.items():
            unique_char_name = f"{book_id}_{char_name}"
            
            if unique_char_name not in master_encode_data:
                master_encode_data[unique_char_name] = []
                
            master_encode_data[unique_char_name].extend(data_list)

    print("\n--- All books encoded! Distilling chunks... ---")
    
    all_results = {
        "ZSHOT": bart_data_dict,
        "ENCODE": master_encode_data
    }
    
    prepare_and_save_chunks(all_results, filename="distill_data_val.pt")
    
if __name__ == "__main__":
    #run from code as root
    #python -m src.test
    #if you are testing bart, use -b, if for pipe (heavy) use -p, foe encode (sbert) use -e
    import argparse
    parser = argparse.ArgumentParser() # No description needed
    parser.add_argument("-p", "--pipe", action="store_true")
    parser.add_argument("-b", "--bart", action="store_true")
    parser.add_argument("-e", "--encode", action="store_true")
    parser.add_argument("-d", "--distill", action="store_true")
    parser.add_argument("-cz", "--cache_zshot", action="store_true")
    args = parser.parse_args()

    if args.bart:
        test_zeroshot_behavioral_pipeline()
    if args.pipe:
        test_pipeline_with_real_data()
    if args.encode:
        test_encoding()
    if args.distill:
        generate_distill_dataset()
    if args.cache_zshot:
        cache_zeroshot_pipeline()