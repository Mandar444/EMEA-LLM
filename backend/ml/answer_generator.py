import re
import json
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional

class AnswerGenerator:
    """
    Offline Extractive QA & Summarization Layer.
    Extracts sentences from matching document chunks, scores them semantically and lexically,
    applies grammar and transition cleanups, de-duplicates semantically similar sentences,
    injects citations, and formats responses dynamically based on query intent.
    """
    def __init__(self, tokenizer, encoder, bm25):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.bm25 = bm25
        self.decorative_headings = {
            "getting started",
            "introduction",
            "overview",
            "purpose",
            "interface overview",
        }
        self.procedural_triggers = {
            "how", "steps", "step", "procedure", "instructions", "instruction",
            "guide", "log in", "login", "configure", "access", "select", "save",
        }
        self.instruction_starters = (
            "enter ", "click ", "select ", "choose ", "fill ", "specify ", "open ",
            "go ", "press ", "use ", "set ", "make ", "confirm ", "upload ",
            "download ", "delete ", "create ", "edit ", "change ", "access ",
            "after ", "this opens", "the system will", "access to ",
        )

        # List of transition words to strip from sentence starts to improve flow
        self.transition_patterns = [
            r'^however,\s*',
            r'^therefore,\s*',
            r'^consequently,\s*',
            r'^as\s+a\s+result,\s*',
            r'^for\s+example,\s*',
            r'^furthermore,\s*',
            r'^moreover,\s*',
            r'^in\s+addition,\s*',
            r'^on\s+the\s+other\s+hand,\s*',
            r'^nevertheless,\s*',
            r'^nonetheless,\s*',
            r'^thus,\s*',
            r'^hence,\s*',
            r'^accordingly,\s*',
            r'^meanwhile,\s*',
            r'^subsequently,\s*',
            r'^in\s+contrast,\s*',
            r'^specifically,\s*'
        ]
        
    def split_sentences(self, text: str) -> List[str]:
        """
        Segments a block of text into distinct sentences using a robust regex.
        Ignores decimal points, file names, and common abbreviations.
        """
        if not text:
            return []
        # Splits on period/question/exclamation followed by space or end of string.
        # Excludes splits after common abbreviations or single letter initials.
        # Split into separate fixed-width look-behinds to prevent compilation error in Python re module.
        sentence_end = re.compile(
            r'(?<!\b[A-Za-z]\.)'
            r'(?<!\b\d\.)'
            r'(?<!\b(?:eq|vs|eg|ie|dr|mr|ms|v4|v3|v2|v1)\.)'
            r'(?<!\b(?:vol)\.)'
            r'(?<=\.|\?|\!)\s+'
        )
        raw_sentences = sentence_end.split(text)
        sentences: List[str] = []

        for raw in raw_sentences:
            value = raw.strip()
            if not value:
                continue

            # Section titles often arrive glued to the first procedural step:
            # "Getting Started ... To log in, perform the following steps: Enter ..."
            # Keep the verbatim actionable tail and drop the decorative preamble.
            step_match = re.search(
                r'\b(?:to\s+[^:]{2,120}|procedure)\s*:\s+',
                value,
                flags=re.IGNORECASE,
            )
            if not step_match:
                step_match = re.search(
                    r'\bto\s+[^.?!:]{2,120}\bsteps\s*:\s+',
                    value,
                    flags=re.IGNORECASE,
                )

            if step_match:
                tail = value[step_match.end():].strip()
                if tail:
                    sentences.extend(self.split_sentences(tail))
                continue

            value = self.strip_decorative_heading_prefix(value)
            if value and not self.is_decorative_heading(value) and not self.is_procedural_preamble(value):
                sentences.append(value)

        return sentences

    def strip_decorative_heading_prefix(self, text: str) -> str:
        """
        Removes short decorative heading prefixes without altering factual sentence text.
        """
        cleaned = text.strip()
        for heading in sorted(self.decorative_headings, key=len, reverse=True):
            pattern = r'^' + re.escape(heading) + r'\s+'
            if re.match(pattern, cleaned, flags=re.IGNORECASE):
                return re.sub(pattern, '', cleaned, count=1, flags=re.IGNORECASE).strip()
        return cleaned

    def is_decorative_heading(self, text: str) -> bool:
        """
        True only for standalone decorative labels, not factual "Purpose of ..." sentences.
        """
        normalized = re.sub(r'[^\w\s-]', '', text).strip().lower()
        return normalized in self.decorative_headings

    def is_procedural_preamble(self, text: str) -> bool:
        """
        Drops non-step lead-ins such as "To log in, perform the following".
        These are useful section scaffolding but not answer content.
        """
        normalized = re.sub(r'\s+', ' ', text).strip().lower().rstrip(".:")
        return bool(re.search(r'\bto\s+[^.?!:]{2,120}\bperform\s+the\s+following$', normalized))

    def is_procedural_query(self, query_text: str) -> bool:
        q = query_text.lower()
        return any(trigger in q for trigger in self.procedural_triggers)

    def is_instructional_sentence(self, text: str) -> bool:
        lowered = text.strip().lower()
        if not lowered or self.is_decorative_heading(lowered):
            return False
        return lowered.startswith(self.instruction_starters)

    def dedupe_candidates_preserve_order(self, candidates: List[Dict[str, Any]], threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Removes exact and near-identical sentences while preserving document order.
        """
        selected: List[Dict[str, Any]] = []
        seen_text = set()
        for cand in candidates:
            normalized = re.sub(r'\s+', ' ', cand["text"].strip().lower())
            if normalized in seen_text:
                continue
            duplicate = False
            if "vector" in cand:
                for existing in selected:
                    if "vector" in existing and float(np.dot(cand["vector"], existing["vector"])) > threshold:
                        duplicate = True
                        break
            if duplicate:
                continue
            seen_text.add(normalized)
            selected.append(cand)
        return selected

    def expand_procedural_neighbors(
        self,
        selected_candidates: List[Dict[str, Any]],
        candidate_sentences: List[Dict[str, Any]],
        max_before: int = 3,
        max_after: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        For procedural answers, include nearby verbatim instructional sentences from
        the same chunk so the answer is a coherent step sequence, not an isolated hit.
        """
        expanded: List[Dict[str, Any]] = []
        selected_keys = {(c["chunk_id"], c["sentence_index"]) for c in selected_candidates}

        for selected in selected_candidates:
            same_chunk = [
                c for c in candidate_sentences
                if c["chunk_id"] == selected["chunk_id"]
            ]
            same_chunk.sort(key=lambda c: c["sentence_index"])
            selected_pos = next(
                (idx for idx, c in enumerate(same_chunk) if c["sentence_index"] == selected["sentence_index"]),
                None,
            )
            if selected_pos is None:
                continue

            start = max(0, selected_pos - max_before)
            end = min(len(same_chunk), selected_pos + max_after + 1)
            for neighbor in same_chunk[start:end]:
                key = (neighbor["chunk_id"], neighbor["sentence_index"])
                is_anchor = key in selected_keys
                if is_anchor or self.is_instructional_sentence(neighbor["text"]):
                    if not self.is_decorative_heading(neighbor["text"]):
                        neighbor["selection_reason"] = (
                            "Selected by sentence rank"
                            if is_anchor
                            else "Included as neighbouring procedural instruction from same chunk"
                        )
                        expanded.append(neighbor)

        expanded.sort(key=lambda c: (c["chunk_index"], c["sentence_index"]))
        return self.dedupe_candidates_preserve_order(expanded)

    def compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Computes L2-normalized Siamese dense embeddings for a list of sentences.
        """
        embeddings = []
        for s in sentences:
            ids = self.tokenizer.encode(s, max_length=120, padding=True)
            tensor_ids = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                vector = self.encoder(tensor_ids).squeeze(0).cpu().numpy()
            embeddings.append(vector)
        return np.array(embeddings, dtype=np.float32)

    def compute_lexical_overlap(self, query_tokens: List[str], sentence_tokens: List[str]) -> float:
        """
        Computes the lexical overlap score of query tokens in the sentence
        weighted by the corpus-wide IDF of each matching word.
        """
        if not query_tokens or not sentence_tokens:
            return 0.0
        
        sent_set = set(sentence_tokens)
        score = 0.0
        for token in query_tokens:
            if token in sent_set:
                # Retrieve IDF of the token from BM25 if available, else default to 1.0
                idf = self.bm25.idf.get(token, 1.0)
                score += idf
        return score

    def clean_sentence_flow(self, text: str) -> str:
        """
        Styles sentence flow by capitalizing initial letters, standardizing terminal punctuation,
        stripping leading markdown markers, and stripping leading transition words.
        """
        if not text:
            return ""
            
        # 1. Strip redundant markdown markers (bullets, headers, hashes, brackets)
        cleaned = re.sub(r'^(?:\d+[\.\)]|[\-\*•#])\s*', '', text).strip()
        
        # 2. Strip leading transition words (case-insensitive)
        for pattern in self.transition_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
            
        if not cleaned:
            return ""
            
        # 3. Capitalize first letter of the cleaned sentence
        cleaned = cleaned[0].upper() + cleaned[1:]
        
        # 4. Standardize terminal punctuation (append period if missing and no punctuation exists)
        if cleaned[-1] not in ['.', '?', '!']:
            cleaned += '.'
            
        return cleaned

    def detect_contradictions(self, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """
        Scans retrieved candidates for contradictory statements.
        Identifies pairs that share high keyword overlap (same topic) but mismatch
        in negation markers or modal permissions (e.g. can vs cannot, admin only vs all users).
        Returns a markdown warning block if a conflict is found.
        """
        negations = {"not", "cannot", "never", "no", "restricted", "denied", "unable", "disabled"}
        admin_words = {"admin", "administrator", "administrators", "admin rights"}
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                s1 = candidates[i]
                s2 = candidates[j]
                
                # Check keyword overlap to ensure they discuss the same topic
                tokens1 = set(w.lower() for w in re.findall(r'\w+', s1["text"]) if len(w) > 4)
                tokens2 = set(w.lower() for w in re.findall(r'\w+', s2["text"]) if len(w) > 4)
                
                common = tokens1.intersection(tokens2)
                # If they share at least 2 significant topic keywords
                if len(common) >= 2:
                    text1_lower = s1["text"].lower()
                    text2_lower = s2["text"].lower()
                    
                    conflict_found = False
                    reason = ""
                    
                    # Check negation mismatch
                    has_neg1 = any(n in text1_lower for n in negations)
                    has_neg2 = any(n in text2_lower for n in negations)
                    
                    if has_neg1 != has_neg2:
                        conflict_found = True
                        reason = "opposite permission or availability statements (can vs. cannot / allowed vs. restricted)"
                        
                    # Check admin rights restriction mismatch
                    has_adm1 = any(a in text1_lower for a in admin_words)
                    has_adm2 = any(a in text2_lower for a in admin_words)
                    
                    if has_adm1 != has_adm2:
                        conflict_found = True
                        reason = "contradictory user role restrictions (administrator-only vs. general user permissions)"
                        
                    if conflict_found:
                        ref1 = f"[{s1.get('citation_index', 1)}] {s1['filename']} (Chunk #{s1['chunk_index']})"
                        ref2 = f"[{s2.get('citation_index', 2)}] {s2['filename']} (Chunk #{s2['chunk_index']})"
                        
                        warning = (
                            "> [!WARNING]\n"
                            "> **Documentation Conflict Detected:** We found potentially contradictory statements in the manual regarding this topic:\n"
                            f"> - According to **{ref1}**: *\"{s1['text']}\"*\n"
                            f"> - According to **{ref2}**: *\"{s2['text']}\"*\n"
                            f"> - **Conflict Category:** {reason}.\n\n"
                        )
                        return warning
        return None

    def generate_title(self, query_text: str) -> str:
        """
        Derives a clean, capitalized title from the user query by stripping
        common question templates and cleaning spaces.
        """
        q = query_text.strip("?").strip()
        q_lower = q.lower()
        for prefix in [
            "how do i ", "how to ", "what is ", "what are ", "why does ", "why do ", 
            "define ", "list of ", "list ", "explain ", "can i ", "can we ", "should we ", "should i ",
            "does the ", "is it "
        ]:
            if q_lower.startswith(prefix):
                q = q[len(prefix):]
                break
        
        # Capitalize words
        title = q.strip().title()
        if not title:
            title = "Information Reference"
        return f"# {title}\n\n"

    def generate_answer(
        self,
        query_text: str,
        query_vector: np.ndarray,
        retrieved_chunks: List[Dict[str, Any]],
        threshold: float,
        fallback_message: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts, ranks, de-duplicates, validates, formats, and cites candidate sentences.
        """
        candidate_sentences: List[Dict[str, Any]] = []
        query_tokens = self.tokenizer.clean_and_split(query_text)
        q_lower = query_text.lower()
        procedural_query = self.is_procedural_query(query_text)
        
        # Define stopwords
        stopwords = {"what", "is", "how", "to", "the", "a", "an", "do", "i", "system", "of", "in", "and", "or", "for", "with", "on", "at", "by", "from", "about", "this", "that", "these", "those", "who", "whom", "whose", "which", "where", "when", "why", "can", "should", "could", "you", "your", "we", "our", "us", "are", "be", "been", "have", "has", "had", "does", "did", "was", "were", "go", "any"}
        query_keywords = [t for t in query_tokens if t not in stopwords and len(t) > 1]
        
        # 1. Segment chunks into sentences and record document offsets
        for chunk in retrieved_chunks:
            filename = chunk.get("filename", "unknown")
            chunk_index = chunk.get("chunk_index", 0)
            document_id = chunk.get("document_id")
            chunk_id = chunk.get("id")
            chunk_score = chunk.get("score", 0.0)
            
            raw_sentences = self.split_sentences(chunk.get("content", ""))
            
            for idx, s in enumerate(raw_sentences):
                # Simple exact match deduplication
                if any(c["text"] == s for c in candidate_sentences):
                    continue
                candidate_sentences.append({
                    "text": s,
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "sentence_index": idx,
                    "chunk_score": chunk_score
                })
                
        # Define actual fallback message
        actual_fallback_message = "I could not find a direct answer to your question in the uploaded documentation."

        if not candidate_sentences:
            return actual_fallback_message, []
            
        # 2. Embed sentences and score them
        sentence_texts = [c["text"] for c in candidate_sentences]
        sentence_vectors = self.compute_sentence_embeddings(sentence_texts)
        
        # Cosine similarity
        semantic_scores = np.dot(sentence_vectors, query_vector)
        
        # Lexical scoring (exclude stopwords)
        lexical_scores = []
        for c in candidate_sentences:
            s_tokens = self.tokenizer.clean_and_split(c["text"])
            lexical_scores.append(self.compute_lexical_overlap(query_keywords, s_tokens))
        
        lexical_scores = np.array(lexical_scores, dtype=np.float32)
        max_lex = np.max(lexical_scores)
        if max_lex > 0:
            norm_lexical_scores = lexical_scores / max_lex
        else:
            norm_lexical_scores = np.zeros_like(lexical_scores)
            
        # Combined score: 0.6 * Lexical + 0.4 * Semantic
        combined_scores = 0.6 * norm_lexical_scores + 0.4 * semantic_scores
        
        for i, c in enumerate(candidate_sentences):
            c["score"] = float(combined_scores[i])
            c["vector"] = sentence_vectors[i]
            c["semantic_score"] = float(semantic_scores[i])
            c["lexical_score"] = float(norm_lexical_scores[i])

        # ----------------------------------------------------
        # Print Debugging Instrumentation (Tasks 1, 2, 3, 4)
        # ----------------------------------------------------
        print("\n" + "="*70)
        print("DEBUG: ANSWER GENERATION PIPELINE INSTRUMENTATION")
        print("="*70)
        print(f"Query: \"{query_text}\"")
        print(f"Query Keywords: {query_keywords}")
        
        print("\n1. Top-K retrieved chunks:")
        for idx, chunk in enumerate(retrieved_chunks):
            print(f"  - Chunk #{chunk.get('chunk_index')} from '{chunk.get('filename')}' | Retrieval Score: {chunk.get('score', 0.0):.4f}")
            print(f"    Snippet: \"{chunk.get('content', '')[:120].strip()}...\"")

        print("\n1a. Extracted sentences by retrieved chunk:")
        grouped = {}
        for c in candidate_sentences:
            grouped.setdefault((c["chunk_id"], c["chunk_index"]), []).append(c)
        for (_, chunk_index), items in grouped.items():
            print(f"  Chunk #{chunk_index}:")
            for c in sorted(items, key=lambda x: x["sentence_index"]):
                print(f"    [{c['sentence_index']}] {c['text']}")
            
        print("\n2. Extracted sentences and similarity scores:")
        for idx, c in enumerate(candidate_sentences):
            emb_str = f"[{c['vector'][0]:.4f}, {c['vector'][1]:.4f}, {c['vector'][2]:.4f}, ...]"
            print(f"  - [{idx}] \"{c['text']}\"")
            print(f"    Embedding Snippet: {emb_str} (dim: {c['vector'].shape[0]})")
            print(f"    Scores -> Semantic: {c['semantic_score']:.4f} | Lexical: {c['lexical_score']:.4f} | Combined: {c['score']:.4f}")
            
        print("\n3. Final ranked sentence list (before validation):")
        ranked_for_log = sorted(candidate_sentences, key=lambda x: x["score"], reverse=True)
        for idx, r in enumerate(ranked_for_log):
            print(f"  Rank {idx+1}: Combined Score: {r['score']:.4f} | Semantic Score: {r['semantic_score']:.4f} | Lexical Score: {r['lexical_score']:.4f} | \"{r['text']}\"")
        # ----------------------------------------------------

        # 3. Filter and Rank Candidate Sentences with Answer Validation
        valid_candidates = []
        validation_logs = []
        for c in candidate_sentences:
            # Keyword match check
            has_keyword = False
            s_tokens_set = set(t.lower() for t in self.tokenizer.clean_and_split(c["text"]))
            if query_keywords:
                has_keyword = any(kw in s_tokens_set for kw in query_keywords)
            else:
                has_keyword = True
                
            # If the sentence has a keyword, it is valid as long as its combined score passes a threshold
            # Otherwise, we discard it to prevent unrelated sentences from contaminating the answer
            if has_keyword:
                is_valid = True
                sem_threshold = -0.5  # Effectively no semantic threshold check for keyword matches
                score_threshold = 0.25
            else:
                is_valid = False  # Strictly discard non-keyword matching sentences
                sem_threshold = 0.50
                score_threshold = 0.32
                
            if is_valid and c["score"] >= score_threshold and c["semantic_score"] >= sem_threshold:
                c["selection_reason"] = "Passed keyword and score validation"
                valid_candidates.append(c)
                validation_logs.append((c["text"], True, "Passed validation filters"))
            else:
                reason = f"Failed cutoff score < {score_threshold}"
                if not is_valid:
                    reason = "Unrelated sentence (no query keywords match)"
                elif c["semantic_score"] < sem_threshold:
                    reason = f"Failed semantic threshold < {sem_threshold:.2f}"
                validation_logs.append((c["text"], False, reason))
                
        # Sort candidates descending by score
        valid_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # 4. De-duplicate candidates (cosine similarity check)
        selected_candidates: List[Dict[str, Any]] = []
        dedup_logs = []
        for cand in valid_candidates:
            redundant = False
            for sel in selected_candidates:
                sim = np.dot(cand["vector"], sel["vector"])
                if sim > 0.85:
                    redundant = True
                    dedup_logs.append((cand["text"], sel["text"], sim))
                    break
            if not redundant:
                cand.setdefault("selection_reason", "Selected by sentence rank")
                selected_candidates.append(cand)
                # Keep up to 5 sentences to maintain structured brevity
                if len(selected_candidates) >= 5:
                    break

        if procedural_query and selected_candidates:
            selected_candidates = self.expand_procedural_neighbors(
                selected_candidates,
                candidate_sentences,
            )

        # ----------------------------------------------------
        # Print Validation & Deduplication logs
        # ----------------------------------------------------
        print("\n4. Answer Validation status:")
        for text, passed, reason in validation_logs:
            status_str = "PASSED" if passed else "DISCARDED"
            print(f"  - [{status_str}] \"{text}\"")
            print(f"    Reason: {reason}")
            
        print("\n5. Deduplication actions:")
        if not dedup_logs:
            print("  No sentences were removed by deduplication.")
        else:
            for removed, kept, sim in dedup_logs:
                print(f"  - Removed: \"{removed}\"")
                print(f"    Reason: Semantically redundant with kept sentence \"{kept}\" (similarity: {sim:.4f} > 0.85)")
                
        print("\n6. Finally selected sentences:")
        for idx, s in enumerate(selected_candidates):
            print(f"  - [{idx+1}] Score: {s['score']:.4f} | Source: {s['filename']} (Chunk #{s['chunk_index']}) | \"{s['text']}\"")
            print(f"    Reason: {s.get('selection_reason', 'Selected by sentence rank')}")
        print("="*70 + "\n")
        # ----------------------------------------------------

        # If no sentences passed validation, build the structured fallback
        if not selected_candidates:
            # Sort all candidates by score to get closest ones
            candidate_sentences.sort(key=lambda x: x["score"], reverse=True)
            closest_candidates = [c for c in candidate_sentences if c["score"] >= 0.22][:2]
            
            if not closest_candidates:
                return actual_fallback_message, []
                
            fallback_sources = []
            source_key_to_cite_idx = {}
            bullets = []
            
            for idx, c in enumerate(closest_candidates):
                src_key = (c["document_id"], c["chunk_index"])
                if src_key not in source_key_to_cite_idx:
                    cite_idx = len(fallback_sources) + 1
                    source_key_to_cite_idx[src_key] = cite_idx
                    fallback_sources.append({
                        "chunk_id": c["chunk_id"],
                        "document_id": c["document_id"],
                        "filename": c["filename"],
                        "chunk_index": c["chunk_index"],
                        "citation_index": cite_idx
                    })
                c["citation_index"] = source_key_to_cite_idx[src_key]
                clean_text = self.clean_sentence_flow(c["text"])
                if clean_text:
                    bullets.append(f"- {clean_text} [{c['citation_index']}]")
            
            if not bullets:
                return actual_fallback_message, []
                
            body = (
                f"{actual_fallback_message}\n\n"
                "### Closest Related Information\n" + "\n".join(bullets)
            )
            
            bibliography = "\n\nSources:\n" + "\n".join(
                f"[{src['citation_index']}] {src['filename']} (Chunk #{src['chunk_index']})"
                for src in fallback_sources
            )
            
            title_header = self.generate_title(query_text)
            return title_header + body + bibliography, fallback_sources
            
        # 5. Classify Query Type and Route Formatting
        is_how = any(word in q_lower for word in ["how", "steps", "procedure", "instructions", "guide"])
        is_what = any(word in q_lower for word in ["what is", "define", "who is", "explain"])
        is_why = any(word in q_lower for word in ["why", "reason"])
        is_list = any(word in q_lower for word in ["list", "items", "features", "categories"])
        is_compare = any(word in q_lower for word in ["compare", "difference", "versus", "vs"])
        is_where = any(word in q_lower for word in ["where", "location", "find", "navigat"])
        is_can_should = any(q_lower.startswith(word) for word in [
            "can ", "should ", "could ", "is it ", "does the ", "do we ", "is there ", "are there "
        ])
        
        # 6. Assign citation indices to contributing chunks
        contributing_sources = []
        source_key_to_cite_idx = {}
        
        for s in selected_candidates:
            src_key = (s["document_id"], s["chunk_index"])
            if src_key not in source_key_to_cite_idx:
                cite_idx = len(contributing_sources) + 1
                source_key_to_cite_idx[src_key] = cite_idx
                contributing_sources.append({
                    "chunk_id": s["chunk_id"],
                    "document_id": s["document_id"],
                    "filename": s["filename"],
                    "chunk_index": s["chunk_index"],
                    "citation_index": cite_idx
                })
            s["citation_index"] = source_key_to_cite_idx[src_key]
            
        # Detect contradictions and compile a warnings block
        contradiction_warning = self.detect_contradictions(selected_candidates)
            
        # Format: "Compare" -> comparison table
        if is_compare:
            sources_data = {}
            for s in selected_candidates:
                src_idx = s["citation_index"]
                clean_text = self.clean_sentence_flow(s["text"])
                if clean_text:
                    if src_idx not in sources_data:
                        sources_data[src_idx] = []
                    sources_data[src_idx].append(f"{clean_text} [{src_idx}]")
            
            rows = []
            for src_idx, facts in sorted(sources_data.items()):
                # Find filename corresponding to src_idx
                fname = next((src["filename"] for src in contributing_sources if src["citation_index"] == src_idx), "Source")
                src_label = f"[{src_idx}] {fname} (Chunk #{next(src['chunk_index'] for src in contributing_sources if src['citation_index'] == src_idx)})"
                rows.append(f"| {src_label} | {' '.join(facts)} |")
            
            table_header = "| Source Reference | Key Extracted Details |\n| :--- | :--- |\n"
            answer_body = table_header + "\n".join(rows)
 
        # Format: "Can / Should" -> Yes/No prefix + explanation
        elif is_can_should:
            negatives = ["not", "cannot", "never", "no", "restricted", "denied", "failed", "unable", "disabled"]
            positives = ["yes", "can", "should", "allow", "permitted", "available", "support", "authorized", "possible", "enabled"]
            
            has_negative = False
            has_positive = False
            for s in selected_candidates:
                s_lower = s["text"].lower()
                if any(word in s_lower for word in negatives):
                    has_negative = True
                if any(word in s_lower for word in positives):
                    has_positive = True
            
            if has_negative and not has_positive:
                prefix = "**No.** "
            elif has_positive and not has_negative:
                prefix = "**Yes.** "
            else:
                prefix = "**Partially / Context Dependent.** "
                
            selected_candidates = selected_candidates[:1]
            selected_candidates.sort(key=lambda x: (x["chunk_index"], x["sentence_index"]))
            parts = []
            for s in selected_candidates:
                clean_text = self.clean_sentence_flow(s["text"])
                if clean_text:
                    parts.append(f"{clean_text} [{s['citation_index']}]")
            answer_body = prefix + " ".join(parts)
 
        # Format: "What" -> short definition (copula trigger priority, return only 1 sentence)
        elif is_what:
            def_triggers = [" is a", " refers to", " is defined as", " means", " represents"]
            best_def = None
            for s in selected_candidates:
                if any(trigger in s["text"].lower() for trigger in def_triggers):
                    best_def = s
                    break
            if not best_def:
                best_def = selected_candidates[0]
                
            clean_text = self.clean_sentence_flow(best_def["text"])
            answer_body = f"{clean_text} [{best_def['citation_index']}]"
            
        # Format: "How" -> numbered steps list
        elif is_how:
            selected_candidates = selected_candidates[:5]
            selected_candidates.sort(key=lambda x: (x["chunk_index"], x["sentence_index"]))
            steps = []
            for idx, s in enumerate(selected_candidates):
                clean_text = self.clean_sentence_flow(s["text"])
                if clean_text:
                    steps.append(f"{idx + 1}. {clean_text} [{s['citation_index']}]")
            answer_body = "\n".join(steps)
            
        # Format: "Why" -> brief explanation
        elif is_why:
            selected_candidates = selected_candidates[:1]
            selected_candidates.sort(key=lambda x: (x["chunk_index"], x["sentence_index"]))
            parts = []
            for s in selected_candidates:
                clean_text = self.clean_sentence_flow(s["text"])
                if clean_text:
                    parts.append(f"{clean_text} [{s['citation_index']}]")
            answer_body = " ".join(parts)
            
        # Format: "List" -> bullet points
        elif is_list:
            selected_candidates = selected_candidates[:3]
            selected_candidates.sort(key=lambda x: (x["chunk_index"], x["sentence_index"]))
            bullets = []
            for s in selected_candidates:
                clean_text = self.clean_sentence_flow(s["text"])
                if clean_text:
                    bullets.append(f"- {clean_text} [{s['citation_index']}]")
            answer_body = "\n".join(bullets)
            
        # Format: "Where" -> navigation breadcrumbs list with location icons
        elif is_where:
            selected_candidates = selected_candidates[:1]
            selected_candidates.sort(key=lambda x: (x["chunk_index"], x["sentence_index"]))
            bullets = []
            for s in selected_candidates:
                clean_text = self.clean_sentence_flow(s["text"])
                if clean_text:
                    bullets.append(f"- 📍 {clean_text} [{s['citation_index']}]")
            answer_body = "\n".join(bullets)
            
        # Default -> concise paragraph of top 3 chronological sentences
        else:
            selected_candidates = selected_candidates[:1]
            selected_candidates.sort(key=lambda x: (x["chunk_index"], x["sentence_index"]))
            parts = []
            for s in selected_candidates:
                clean_text = self.clean_sentence_flow(s["text"])
                if clean_text:
                    parts.append(f"{clean_text} [{s['citation_index']}]")
            answer_body = " ".join(parts)
            
        # Prepend contradiction warnings block if one was detected
        if contradiction_warning:
            answer_body = contradiction_warning + answer_body
            
        # 7. Append Bibliography Footer
        bibliography = "\n\nSources:\n" + "\n".join(
            f"[{src['citation_index']}] {src['filename']} (Chunk #{src['chunk_index']})"
            for src in contributing_sources
        )
        
        # 8. Prepend Auto-Generated Title
        title_header = self.generate_title(query_text)
        final_answer = title_header + answer_body + bibliography
        return final_answer, contributing_sources
