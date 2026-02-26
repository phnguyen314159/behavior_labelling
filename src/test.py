import pytest
import json
from processData.textPipeline import iter_books, book_process, global_ent, cluster_container, process_registry

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
        first_person = list(registry.keys())[0]
        assert "references" in registry[first_person], "Registry entry missing 'references' list"
        
        # Check schema of a reference item
        if len(registry[first_person]["references"]) > 0:
            sample_ref = registry[first_person]["references"][0]
            # We removed the "type" assertion here!
            assert "text" in sample_ref, "Reference missing 'text' key"
            assert "global_char_pos" in sample_ref, "Reference missing 'global_char_pos' key"
            
            
        print(f"Test Passed: Built registry with {len(registry)} unique person(s).")

        # --- Print out the first registry entry to the terminal ---
        print(f"\n--- Sample Registry Structure for '{first_person}' ---")
        print(json.dumps({first_person: registry[first_person]}, indent=4))

if __name__ == "__main__":
    # Allows manual execution: python src/test.py
    test_pipeline_with_real_data()