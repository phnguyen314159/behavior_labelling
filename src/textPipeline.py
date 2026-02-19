from pathlib import Path
from collections import Counter, defaultdict

import spacy
spacy.prefer_gpu()
from fastcoref import spacy_component
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'device': 'cuda'})

def sliding_window(text, window_size=20, step=17):
    lines = text.splitlines(keepends=True)
    for i in range(0, len(lines), step):
        chunk_text = "".join(lines[i : i + window_size])

        # Calculate global char offset
        char_offset = sum(len(line) for line in lines[:i])

        # Metadata
        line_metadata = {
            "start_line": i,
            "end_line": min(i + window_size, len(lines))
        }

        context = {"offset": char_offset, "lines": line_metadata}

        yield (chunk_text, context)

global_ent = []
book_container = [] #propose for book id
def book_process(text):
    doc_container = []

    #Rename 'offset' to 'context' because it contains the whole dict
    for doc, context in nlp.pipe(sliding_window(text, window_size=20, step=17), as_tuples=True): #add key to here too in front of in - completed

        # Save BOTH doc and context as a tuple.
        # This prepares the data for the Registry step without needing a re-run.
        doc_container.append((doc, context))

        doc_id = len(doc_container) - 1 # doc

        # Extract the actual integer offset from the context dictionary
        chunk_start_offset = context["offset"]

        # we start fill the container for entity
        # FIX: The 'if' check must be INSIDE the loop, or checking 'ent' before defining it will fail
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                global_ent.append({
                    "type": "PERSON",
                    "text": ent.text,
                    "global_start": ent.start_char + chunk_start_offset, #key char pos of the 1st char of the ent

                    #global end - completed
                    "global_end": ent.end_char + chunk_start_offset,

                    "doc_id": doc_id,

                    #try to figure out the sentence (within the doc) that the entity is in! - completed
                    "sentence_text": ent.sent.text,

                    #I THINK THERE IS A SPACY METHOD TO GET THE line THAT CONTAIN THIS ENT - completed
                    # Logic: We take the chunk's start line index and add the number of newlines found before the entity
                    "line_id": context["lines"]["start_line"] + doc.text[:ent.start_char].count('\n')
                })

        #coref
        if doc._.coref_clusters:
            for cluster in doc._.coref_clusters:
                # SAFETY CHECK 1: Ensure the cluster list itself isn't None
                if cluster is None:
                    continue

                # SAFETY CHECK 2: Iterate item by item instead of unpacking immediately
                for item in cluster:
                    if item is None:
                        continue # Skip this specific item if it is None (prevents the crash)

                    start, end = item # Now should beis safe to unpack

                    # Map local token indices to global character offsets
                    span = doc.char_span(start, end)
                    if span:
                        global_ent.append({
                            "type": "COREF",
                            "text": span.text,
                            "global_start": span.start_char + chunk_start_offset, #keys

                            #get global_end: start + len - completed
                            "global_end": span.end_char + chunk_start_offset,

                            #is it possible to get sentence/line id? - completed
                            "sentence_text": span.sent.text,
                            "line_id": context["lines"]["start_line"] + doc.text[:span.start_char].count('\n'),

                            "doc_id": doc_id
                            #try to figure out the sentence (within the doc) that the coref is in! - completed
                            # (Already handled by span.sent.text above)
                        })

        # Clear RAM
        doc._.trf_data = None

