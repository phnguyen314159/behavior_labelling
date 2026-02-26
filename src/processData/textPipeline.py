import os
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

# 1. Bypass the Microsoft STL version check (STL1002) and the unsupported version check
# We pass -Xcompiler to tell nvcc to send the /D macro directly to the C++ compiler
os.environ["NVCC_APPEND_FLAGS"] = "-allow-unsupported-compiler -Xcompiler /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

# 2. Maintain your existing path injections
msvc_bin = os.environ.get("MSVC_PATH")
if msvc_bin:
    msvc_bin = os.path.normpath(msvc_bin) 
    os.environ["PATH"] = msvc_bin + os.path.pathsep + os.environ.get("PATH", "")

if "CUDA_PATH" in os.environ:
    os.environ["PATH"] = os.path.join(os.environ["CUDA_PATH"], "bin") + os.path.pathsep + os.environ["PATH"]

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

def sliding_window(sent_nlp, text, window_size=7, step=5):
    
    sSpans = sentenizer(text, sent_nlp)

    for i in range(0, len(sSpans), step):
        j = min(i + window_size, len(sSpans))
        if i >= j:
            break
        chunk_start = sSpans[i][0] #start of 1st sent, treat like global offset
        chunk_end   = sSpans[j-1][1] #end of 20th sent
        chunk_text  = text[chunk_start:chunk_end] #chunk string
        

        local_spans = [(s - chunk_start, e - chunk_start) for (s, e) in sSpans[i:j]] #list that have the span of each sentence in the chunk, but referencing the chunk itself

        # Metadata
        is_last = (j == len(sSpans))

        context = {
            "offset": chunk_start, 
            "sent_range": {"start_sent": i, "end_sent": j},
            "local_sent_spans": local_spans,
            "is_last": is_last    #add bool to check if last chunk
        }
        
        yield (chunk_text, context)

def get_local_sent_idx(pos: int, local_sent_spans: list[tuple]):
    last_idx = len(local_sent_spans) - 1
    for idx, (s, e) in enumerate(local_sent_spans):
        if s <= pos < e:
            return -1 if idx == last_idx else idx
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

def token_person_ent(tok: Token): #helper for parent finding
    #Return the PERSON entity span containing tok, else None.
    if tok.ent_type_ != "PERSON":
        return None
    # find the actual entity span
    for ent in tok.doc.ents:
        if ent.label_ == "PERSON" and ent.start <= tok.i < ent.end:
            return ent
    return None

def check_depend(doc: Doc, start: int, end: int, max_layers: int = 3): #parents finding
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

#containers: doc_container is a list of tuple (doc, metadata)
#global_ent: list of all 
def book_process(text):
    doc_container = []

    #Rename 'offset' to 'context' because it contains the whole dict
    for doc, context in nlp.pipe(sliding_window(sent_nlp, text), as_tuples=True): #add key to here too in front of in - completed

        # Save BOTH doc and context as a tuple.
        # This prepares the data for the Registry step without needing a re-run, since doc get overwrite with each pipe run
        doc_container.append((doc, context))

        doc_id = len(doc_container) - 1 

        #extract the actual integer offset from the context dictionary
        chunk_start_offset = context["offset"]

        #we start working on the doc imediately to take avantage of its currently being load on living memory so we can take adv of doc_id
        is_last = context.get("is_last", False)
        is_first = (doc_id == 0)
        for ent in doc.ents:
            cur_sent = get_local_sent_idx(ent.start_char, context["local_sent_spans"])
            sent_valid = (cur_sent is not None) and ((is_first and (0 <= cur_sent < 7)) or (is_last and (cur_sent > 0 or cur_sent == -1)) or (not is_first and not is_last and (0 < cur_sent < 7)))
            if ent.label_ == "PERSON" and sent_valid:
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
            #cluster = {(0,1), (4,7), etc)}
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
                cur_sent = get_local_sent_idx(primary[0], context["local_sent_spans"])
                sent_valid = (cur_sent is not None) and ((is_first and (0 <= cur_sent < 7)) or (is_last and (cur_sent > 1 or cur_sent == -1)) or (not is_first and not is_last and (1 < cur_sent < 7)))
                if sent_valid:
                    buffer_ent = check_depend(doc, primary[0], end)
                else:
                    buffer_ent = None
                        
                #all we care is: for doc of id x, what clusters it has, and what is the primary of that cluster (tuple position)
                if buffer_ent != None:
                    for ent in global_ent:
                        if ent["doc_id"] == doc_id and ent["doc_token_pos"] == (buffer_ent.start, buffer_ent.end):
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

'''
#TODO: CHECK CODE, need to fix this into method fittable to run with current pipeline
#This code requirements:  
# 1.fuzzy the global_ent to get a list of unique person we can use (5-7 is enough), but they must be unique
#   i.run a simple 
# 2.for each of the unique person we get: 
#   i.use fuzzy to check global_ent against each unique person |||| example logic: for i, ent in enumerated(global_ent): if fuzzyFunct(ent.get("text")) == true: add index to that unique name bucket under list "ent"
#   ii.for the ent above, check if we have "child_cluster" in ent, if they do, use the same above logic but with [cluster_id and ent's doc_id] to get the index from cluster_container, and add that index to this unique name bucket in list "cluster"
# 3.once we get registry done, we will proceed to work with dups:
#   i.for each unique person bucket, make 2 sets: ent_index and cluster_index
#   ii.for each of the 2 set:
#       I.compare the set between 2 unique person using intersection of sets
#       II.for simplicity, if intersect in ent_index at all, we will merge those 2 unique person bucket to create a new entry in the registry
#       III.meanwhile, if cluster set intersect pass a certain percentage, say 40-50? we also merge the 2 unique person bucket in the registry


what i want register to looks like:
registry = {
    "Adder": {
        "references": [
            {
                "global_start": 10452,  # Original offset in the .txt
                "text": "he",            # Pronoun reference
                "doc_id": 17,           # Integer index for memory safety
                "local_line": 5,        # Sentence index in this chunk
                "local_span": [15, 17], # Position in the current doc
                "type": "COREF"         # Categorized as coreference/pronoun
            },
            {
                "global_start": 10440,  
                "text": "Adder",
                "doc_id": 17,
                "local_line": 4,
                "local_span": [3, 8],
                "type": "PERSON"        # Primary NER extraction
            },
            {
                "global_start": 10510,  # A new alias mention further in the text
                "text": "Mr. Black",    # The specific alias you requested
                "doc_id": 17,
                "local_line": 8,
                "local_span": [22, 31],
                "type": "PERSON"        # Treat aliases as PERSON for NER logic
            }
        ]
    }
}
'''

def fuzzy_match(s1, s2, threshold=0.8):
    """Helper for fuzzy string matching."""
    return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio() >= threshold

def process_registry(global_ent, cluster_container):
    # 1. fuzzy the global_ent to get a list of unique person we can use (5-7 is enough)
    name_counts = Counter([ent["text"].strip() for ent in global_ent])
    unique_persons = []
    
    for name, _ in name_counts.most_common():
        if len(unique_persons) >= 7:
            break
        # 1.i. run a simple check to ensure they are unique
        if not any(fuzzy_match(name, up) for up in unique_persons):
            unique_persons.append(name)

    # 2. for each of the unique person we get:
    # Build buckets holding sets of indices: { "Adder": {"ent": set(), "cluster": set()} }
    buckets = {up: {"ent": set(), "cluster": set()} for up in unique_persons}
    
    for i, ent in enumerate(global_ent):
        ent_text = ent["text"].strip()
        
        # 2.i. use fuzzy to check global_ent against each unique person
        for up in unique_persons:
            if fuzzy_match(ent_text, up):
                # add index to that unique name bucket under list "ent"
                buckets[up]["ent"].add(i)
                
                # 2.ii. check if we have "child_cluster" in ent
                if "child_cluster" in ent:
                    c_id = ent["child_cluster"]
                    d_id = ent["doc_id"]
                    
                    # use [cluster_id and ent's doc_id] to get the index from cluster_container
                    for j, cluster in enumerate(cluster_container):
                        if cluster["cluster_id"] == c_id and cluster["doc_id"] == d_id:
                            # add that index to this unique name bucket in list "cluster"
                            buckets[up]["cluster"].add(j)
                break # Move to next ent once a bucket is found

    # 3. once we get registry done, we will proceed to work with dups
    merged_registry = []
    
    for up, data in buckets.items():
        # 3.i. for each unique person bucket, make 2 sets: ent_index and cluster_index
        ent_set = data["ent"]
        cluster_set = data["cluster"]
        
        has_merged = False
        # 3.ii. for each of the 2 set:
        for mb in merged_registry:
            # 3.ii.I. compare the set between 2 unique person using intersection of sets
            ent_intersect = len(ent_set.intersection(mb["ent"])) > 0
            
            # 3.ii.III. meanwhile, if cluster set intersect pass a certain percentage (40%)
            cluster_intersect = False
            if len(cluster_set) > 0 and len(mb["cluster"]) > 0:
                overlap = len(cluster_set.intersection(mb["cluster"]))
                min_len = min(len(cluster_set), len(mb["cluster"]))
                if (overlap / min_len) >= 0.4:
                    cluster_intersect = True
                    
            # 3.ii.II. if intersect in ent_index at all (or cluster >= 40%), merge those 2
            if ent_intersect or cluster_intersect:
                mb["ent"].update(ent_set)
                mb["cluster"].update(cluster_set)
                
                # Keep the longer, more descriptive name for the registry key
                if len(up) > len(mb["name"]):
                    mb["name"] = up
                    
                has_merged = True
                break
                
        if not has_merged:
            merged_registry.append({
                "name": up,
                "ent": ent_set,
                "cluster": cluster_set
            })

    # Build the final output dictionary
    registry = {}
    
    for mb in merged_registry:
        primary_name = mb["name"]
        references = []
        
        # Populate PERSON mentions
        for e_idx in mb["ent"]:
            ent = global_ent[e_idx]
            references.append({
                "global_char_pos": ent["global_start"],
                "text": ent["text"],
                "doc_ptr": f"<spacy_doc_{ent['doc_id']}>",
                "local_line": ent.get("sentence_id", -1),
                "local_span": list(ent["doc_token_pos"])
            })
            
        # skipping the detailed pronoun texts for now
        # including the structure based ONLY on the existing cluster_container data
        for c_idx in mb["cluster"]:
            clust = cluster_container[c_idx]
            references.append({
                "global_char_pos": -1, # Original code doesn't save global start for clusters
                "text": "COREF_PRIMARY", 
                "doc_ptr": f"<spacy_doc_{clust['doc_id']}>",
                "local_line": -1, 
                "local_span": list(clust["primary"])
            })
            
        registry[primary_name] = {"references": references}
        
    return registry