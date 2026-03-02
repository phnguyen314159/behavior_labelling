from spacy import tokens
from collections import deque
from src.config import WINDOW_SIZE, STEP, LAST_INDEX, BATCH_SIZE
from processData.textPipeline import doc_container, registry

def clean_persp(target_name, current_doc_id, doc_container, registry):
    doc, context = doc_container[current_doc_id]
    is_last = context.get("is_last", False)
    is_first = (current_doc_id == 0)
    
    # 1. Standard Registry Filter
    doc_refs = [ref for ref in registry[target_name]["references"] if ref["doc_id"] == current_doc_id]

    from collections import defaultdict
    sent_map = defaultdict(list)
    for ref in doc_refs:
        sent_map[ref["local_line"]].append(ref)

    
    
    for sent_idx, sent in enumerate(doc.sents):
        if is_first:
            sent_valid = (0 <= sent_idx < LAST_INDEX)
        elif is_last:
            sent_valid = (sent_idx > 0)
        else:
            sent_valid = (0 < sent_idx < LAST_INDEX)

        # Only perform the surgery if the sentence belongs to this chunk's "unique core"
        if not sent_valid:
            continue

        # 2. FIND CLAUSE HEADS (The new Clause-Level Prefix logic)
        clause_injections = []
        for token in sent:
            # We look for the start of dependent or coordinate clauses
            if token.dep_ in ["advcl", "xcomp", "conj"]:
                # Check 3-breath ancestors to see if it belongs to Target
                if any(a.i in [r["global_start"] for r in doc_refs] for a in list(token.ancestors)[:3]):
                    clause_injections.append({
                        "start": token.idx, # token.idx is spacy's method to get char pos for token
                        "label": f"[{target_name}'s clause] "
                    })

        # 3. get doc-wise position
        sent_text = sent.text
        sent_start_offset = sent.start_char
        
        # Merge your registry refs with your new clause injections
        all_edits = []
        
        # Add registry mentions
        for ref in sent_map[sent_idx]:
            is_poss = ref.get("text", "").lower() in ["his", "her", "its", "their"] or ref.get("type") == "PRP$"
            all_edits.append({
                "start": ref["local_span"][0],
                "label": f"[{target_name}'s] " if is_poss else f"[{target_name}] "
            })
            
        # Add clause prefixes
        all_edits.extend(clause_injections)

        # 4. APPLY EDITS (Reverse order so we dont have to manually keep track of char_pos shifting
        sorted_edits = sorted(all_edits, key=lambda x: x["start"], reverse=True)
        
        for edit in sorted_edits:
            s_local = edit["start"] - sent_start_offset
            # Safety: Ensure s_local is within sentence bounds
            if 0 <= s_local <= len(sent_text):
                sent_text = sent_text[:s_local] + edit["label"] + sent_text[s_local:]

        yield sent_idx, sent_text
    return None

            
def scene_prep_generator(doc_container, registry, target_name):
    sentence_queue = deque(maxlen=4)
    prompt_prefix = f"Task: Analyze {target_name}. Context: "

    for doc_id, (doc, context) in enumerate(doc_container):
        is_last = context.get("is_last", False)
        is_first = (doc_id == 0)
        all_raw_sents = list(doc.sents)
        
        # 1. INITIALIZE: Engage generator for the first milestone in this doc
        sentence_gen = clean_persp(target_name, doc_id, doc_container, registry)
        
        def get_next_f():
            try:
                # Returns (f_idx, f_text) from your generator
                return next(sentence_gen)
            except StopIteration:
                return WINDOW_SIZE, None

        f_idx, f_text = get_next_f()

        # 2. INNER LOOP: Chronological s_idx through the doc sentences
        for s_idx, sent in enumerate(doc.sents):
            # Deduplication and edge case Check
            if is_first:
                sent_valid = (0 <= s_idx < LAST_INDEX)
            elif is_last:
                sent_valid = (s_idx > 0)
            else:
                sent_valid = (0 < s_idx < LAST_INDEX)

            if not sent_valid:
                continue

            # 3. PUSH LOGIC: Compare s_idx against f_idx
            if s_idx < f_idx:
                # Keep pushing raw s_idx until we hit the milestone
                sentence_queue.append(sent.text)
            
            elif s_idx == f_idx:
                # MILESTONE REACHED: Push f_idx
                sentence_queue.append(f_text)
                next_idx = s_idx + 1
                # RE-ENGAGE: Immediately get new f_idx for the next milestone
                
                
                # 4. THE "LAST CHECK": Push one more to complete the context
                # We peek at s_idx + 1 to provide the 'Post' context
                # edge case (last chunk), for all other cases we know because of dedup above we can do future peak, but not for last chunk where we let it iter to len-1
                # we know that a chunk must always <WINDOW_SIZE, therefor we can easily know if we are at the very last line
                if next_idx < WINDOW_SIZE: 
                    f_idx, f_text = get_next_f()
                    
                    # If the very next line is also a milestone, push its fixed version
                    if next_idx == f_idx:
                        sentence_queue.append(f_text)
                            # Re-engage again because we just consumed this milestone
                        f_idx, f_text = get_next_f()
                    else:
                            # Otherwise, push the raw s_idx+1 neighbor
                        sentence_queue.append(all_raw_sents[next_idx].text)

                final_chunk = " ".join(list(sentence_queue)).strip()
                yield f"{prompt_prefix}{final_chunk}", (s_idx + doc_id*WINDOW_SIZE) #critical: gen will now yield both the scene string AND the id of the curr line, for temporal

def scene_batch_gen(doc_container, registry, M=BATCH_SIZE):
    # Initialize character-specific queues
    queues = {name: [] for name in registry.keys()}
    active_generators = {
        name: scene_prep_generator(doc_container, registry, name) 
        for name in registry.keys()
    }

    while active_generators:
        for name in list(active_generators.keys()):
            try:
                # 1. Busy Work: Pull the next scene for this character lens
                text, sent_idx = next(active_generators[name])
                queues[name].append((sent_idx, text))
                
                # 2. Threshold Check: If queue is size M, yield the batch
                if len(queues[name]) == M:
                    yield name, queues[name] # Yield the batch then clean queue
                    queues[name] = []
                    
            except StopIteration:
                # 3. Final Flush: Child generator died, yield remaining scenes
                if queues[name]:
                    yield queues[name]
                del active_generators[name]
                del queues[name]