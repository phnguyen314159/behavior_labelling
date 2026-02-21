from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterator, Tuple, Literal
from spacy.tokens import Doc, Span, Token
import difflib
import spacy
spacy.prefer_gpu()
from fastcoref import spacy_component
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'device': 'cuda'})



def make_sentencizer():
    sent_nlp = spacy.blank("en")
    sent_nlp.add_pipe("sentencizer")
    return sent_nlp

sent_nlp = make_sentencizer() #global

def sentenizer(text, sent_nlp): 
    sent_nlp.max_length = max(sent_nlp.max_length, len(text) + 1)
    doc = sent_nlp(text)
    sent_spans = [(s.start_char, s.end_char) for s in doc.sents]
    return sent_spans

#call with a folder in mind, generator for all file in the folder
Mode = Literal["train", "test", "validation"]
def iter_books(mode: Mode, base_dir: str | Path = "data/book", pattern: str = "*.clean.txt"):
    """
    Yields (book_id, text) for each .txt file in:
      data/book/<mode>/
    where mode is one of: train, test, validation
    """
    data_dir = Path(base_dir) / mode

    if not data_dir.exists():
        raise FileNotFoundError(f"Folder not found: {data_dir.resolve()}")

    for fp in sorted(data_dir.glob(pattern)):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        book_id = fp.name.split(".")[0]
        yield book_id, text

def sliding_window(sent_nlp, text, window_size=20, step=17):
    
    sSpans = sentenizer(text, sent_nlp)

    for i in range(0, len(sSpans), step):
        j = min(i + window_size, len(sSpans))
        if i >= j:
            break
        chunk_start = sSpans[i][0] #start of 1st sent
        chunk_end   = sSpans[j-1][1] #end of 20th sent
        chunk_text  = text[chunk_start:chunk_end]

        local_spans = [(s - chunk_start, e - chunk_start) for (s, e) in sSpans[i:j]]

        # Metadata
        context = {
            "offset": chunk_start, 
            "sent_range": {"start_sent": i, "end_sent": j},
            "local_sent_spans": local_spans    
        }

        yield (chunk_text, context)

def get_local_sent_idx(pos: int, local_sent_spans: list[tuple]):
    for idx, (s, e) in enumerate(local_sent_spans):
        if s <= pos < e:
            return idx
    return None  # not found

def score_mention(span):
    """
    Heuristic scoring for a coref mention span.
    - PROPN: +5 (+10 extra if long)
    - PRON: +3
    - else: +1
    - length bonus: +min(10, len(span.text))
    """
    root_pos = span.root.pos_
    n_chars = len(span.text.strip())

    score = 0
    if root_pos == "PROPN":
        score += 5
        if n_chars >= 6:   # "long propn" threshold; tweak as needed
            score += 5
    elif root_pos == "PRON":
        score += 3
    else:
        score += 1

    score += min(5, n_chars)  # small length tie-breaker
    return score


global_ent = []
book_container = [] #propose for book id
cluster_container = []
registry = defaultdict(lambda: {"references": []})

def token_person_ent(tok: Token):
    #Return the PERSON entity span containing tok, else None.
    if tok.ent_type_ != "PERSON":
        return None
    # find the actual entity span
    for ent in tok.doc.ents:
        if ent.label_ == "PERSON" and ent.start <= tok.i < ent.end:
            return ent
    return None

def check_depend(doc: Doc, start: int, end: int, max_layers: int = 3):
    """
    Start from a char span (start_char:end_char). Take its root token and climb head->head...
    until you hit a PERSON entity, or run out of layers.

    Returns:
      (person_entity_span_or_None, token_where_found_or_None, layers_used)
    """
    span = doc.char_span(start, end, alignment_mode="expand")
    if span is None:
        return None

    tok = span.root
    layers = 0

    while layers < max_layers:
        ent = token_person_ent(tok)
        if ent is not None:
            return ent

        if tok.head is None or tok.head == tok:  # reached sentence root
            return None

        tok = tok.head
        layers += 1
    return None

def book_process(text):
    doc_container = []

    #Rename 'offset' to 'context' because it contains the whole dict
    for doc, context in nlp.pipe(sliding_window(sent_nlp, text, window_size=20, step=17), as_tuples=True): #add key to here too in front of in - completed

        # Save BOTH doc and context as a tuple.
        # This prepares the data for the Registry step without needing a re-run, since doc get overwrite with each pipe run
        doc_container.append((doc, context))

        doc_id = len(doc_container) - 1 

        #extract the actual integer offset from the context dictionary
        chunk_start_offset = context["offset"]

        #we start working on the doc imediately to take avantage of its currently being load on living memory so we can take adv of doc_id
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                global_ent.append({
                    "type": "PERSON",
                    "text": ent.text,
                    "global_start": ent.start_char + chunk_start_offset, #key char pos of the 1st char of the ent
                    "global_end": ent.end_char + chunk_start_offset,

                    "doc_id": doc_id,
                    "doc_token_pos": (ent.start, ent.end), #later we can access using span = doc[tok_pos[0]:tok_pos[1]]
                    "sentence_id": get_local_sent_idx(ent.start_char, context["local_sent_spans"])
                })

        #coref
        #redo: now for each cluster, decide the primary person, and store in temp container
        if doc._.coref_clusters:
            for cluster_id, cluster in enumerate(doc._.coref_clusters):
                # SAFETY CHECK 1: Ensure the cluster list itself isn't None
                if cluster is None:
                    continue
                buffer = 0
                # SAFETY CHECK 2: Iterate item by item instead of unpacking immediately
                for item in cluster:
                    if item is None:
                        continue # Skip this specific item if it is None (prevents the crash)

                    start, end = item # Now should be safe to unpack

                    # Map local token indices to global character offsets
                    span = doc.char_span(start, end)
                    score = score_mention(span)
                    if buffer < score:
                        buffer = score
                        primary = (start, end)
                    # we only record link if primary is out of buffer zone to avoid checking primary in 2 different doc but actually same word, we can just link that using overlapping cluster
                    #fix, streamline child_cluster linking directly into global_ent for easier access
                    cur_sent = get_local_sent_idx(start, context["local_sent_spans"])
                    if 1 < cur_sent < 19:
                        buffer_ent = check_depend(doc, start, end)
                    else:
                        buffer_ent = None
                        
                #all we care is: for doc of id x, what clusters it has, and what is the primary of that cluster (tuple position)
                if buffer_ent != None:
                    for ent in global_ent:
                        if ent["doc_id"] == doc.id and ent["doc_token_pos"] == (buffer_ent.start, buffer_ent.end):
                        # Inject the new data element into 
                            ent["child_cluster"] = cluster_id
   
                cluster_container.append({
                    "doc_id": doc_id,
                    "cluster_id": cluster_id,
                    "primary": primary
                })

        # Clear RAM
        try:
            doc._.trf_data = None
        except Exception:
            pass
    return doc_container


#TODO: CHECK CODE, need to fix this into method fittable to run with current pipeline
#This code requirements:  
# 1.fuzzy the global_ent to get a list of unique person we can use (5-7 is enough), but they must be unique
#   i.run a simple 
# 2.for each of the unique person we get: 
#   i.use fuzzy to check global_ent against each unique person |||| example logic: for i, ent in enumerated(global_ent): if fuzzyFunct(ent.get("text")) == true: add index to that unique name bucket under list "ent"
#   ii.for the ent above, check if we have "child_cluster" in ent, if they do, use the same above logic but with cluster_id to get the index from cluster_container, and add that index to this unique name bucket in list "cluster"
# 3.once we get registry done, we will proceed to work with dups:
#   i.for each unique person bucket, make 2 sets: ent_index and cluster_index
#   ii.for each of the 2 set:
#       I.compare the set between 2 unique person using intersection of sets
#       II.for simplicity, if intersect in ent_index at all, we will merge those 2 unique person bucket to create a new entry in the registry
#       III.meanwhile, if cluster set intersect pass a certain percentage, say 40-50? we also merge the 2 unique person bucket in the registry

# 1. To track counts: { "Entity Name": total_mentions }
entity_popularity = Counter()
# 2. To track clusters: { "Entity Name": {set of unique mention texts} }
entity_mentions_library = {}

def get_canonical_name(new_name, existing_names, cutoff=0.8):
    """
    Check if 'new_name' is similar enough to an existing name.
    Returns the existing name if found, otherwise None.
    """
    # specific override for exact substring matches (e.g. "Harry" -> "Harry Potter")
    for name in existing_names:
        if new_name in name or name in new_name:
            # Return the longer/more complete version as the canonical name
            return name if len(name) > len(new_name) else new_name

    # Fuzzy match for typos
    matches = difflib.get_close_matches(new_name, existing_names, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    return None

# FIX: Unpack the tuple (doc, context) here!
for doc_id, (doc, context) in enumerate(doc_container):

    # FastCoref results are in doc._.coref_clusters
    if not doc._.coref_clusters:
        continue

    for cluster in doc._.coref_clusters:
        # Safety Check: Ensure the cluster list itself isn't None
        if cluster is None:
            continue

        # Find the "Main Name" (the NER entity) inside this cluster
        main_name = None
        cluster_mentions = []

        # we iterate item by item to prevent crash on None
        for item in cluster:
            # Safety Check: STOP THE CRASH
            if item is None:
                continue

            # Now it is safe to unpack
            start, end = item

            span = doc.char_span(start, end)
            if not span: continue

            cluster_mentions.append(span.text)

            # Check if this specific mention is also an NER Entity
            # Optimization: Only check if we haven't found a main name yet
            if not main_name:
                for ent in doc.ents:
                    if ent.start_char == span.start_char and ent.label_ == "PERSON":
                        main_name = ent.text
                        break

        # If we found a Person's name in the cluster, update our counts
        if main_name:
            # TODO: regex! we want dups and dups with close similarity
            # Regex finds exact patterns (e.g., "all emails"), while Fuzzy Matching finds similar text (e.g., "Jonh" ≈ "John").

            # Regex is avoided because names vary by spelling and completeness (e.g., typos, partials), not by a strict structural pattern.
            # https://medium.com/@m.nath/fuzzy-matching-algorithms-81914b1bc498 fuzzy matching algorithms
            # Logic: Check if this main_name is actually a variation of someone we already have

            existing_names = list(entity_popularity.keys())

            # Helper function determines if 'Harry' == 'Harry Potter'
            canonical = get_canonical_name(main_name, existing_names)

            if canonical:
                # Merge this count into the existing one
                final_name = canonical
            else:
                # This is a new unique person
                final_name = main_name

            # Update counts
            entity_popularity[final_name] += len(cluster)

            # Update library
            if final_name not in entity_mentions_library:
                entity_mentions_library[final_name] = set()
            entity_mentions_library[final_name].update(cluster_mentions)
