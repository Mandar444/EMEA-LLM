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
        self.weak_evidence_message = "I couldn't find sufficient information in the uploaded documents to answer this question."

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

    def get_query_keywords(self, query_tokens: List[str]) -> List[str]:
        """
        Keeps only content-bearing query tokens for answer evidence checks.
        Intent words such as "find" should not make an unrelated sentence valid.
        """
        stopwords = {
            "what", "is", "how", "to", "the", "a", "an", "do", "i", "system", "of",
            "in", "and", "or", "for", "with", "on", "at", "by", "from", "about",
            "this", "that", "these", "those", "who", "whom", "whose", "which",
            "where", "when", "why", "can", "should", "could", "you", "your",
            "we", "our", "us", "are", "be", "been", "have", "has", "had",
            "does", "did", "was", "were", "go", "any", "find", "located",
            "location", "tell", "me", "please", "show", "get", "give",
            "information", "info", "details", "document", "documents", "section",
            "page", "module", "field", "item", "explain", "describe", "summarize",
            "complete", "process", "happens", "happen", "walk", "through",
            "shown", "displayed", "display", "displays", "contain", "contains",
            "which", "purpose",
        }
        keywords = []
        for token in query_tokens:
            token = token.lower()
            if token not in stopwords and len(token) > 1:
                if token in {"login", "logging"}:
                    token = "log"
                elif token in {"saving", "saved", "saves"}:
                    token = "save"
                elif token in {"importing", "imported", "imports"}:
                    token = "import"
                elif token in {"signing", "signed", "signs"}:
                    token = "sign"
                elif token in {"navigating", "navigate"}:
                    token = "navigation"
                if token not in keywords:
                    keywords.append(token)
        return keywords

    def is_definition_query(self, query_text: str) -> bool:
        q = query_text.lower().strip()
        return q.startswith(("what is", "define", "explain what", "purpose of"))

    def is_attribute_query(self, query_text: str) -> bool:
        q = query_text.lower()
        return (
            "what information" in q
            or "what is shown" in q
            or "what does" in q and "contain" in q
            or "which information" in q
            or "display" in q
        )

    def is_audit_outcome_query(self, query_text: str) -> bool:
        q = query_text.lower()
        return any(term in q for term in ["discrepancy", "discrepancies", "audit result", "audit status"])

    def is_procedural_intent_query(self, query_text: str) -> bool:
        q = query_text.lower().strip()
        return q.startswith(("how", "explain", "describe", "walk me through"))

    def extract_query_entity(self, query_text: str) -> str:
        q = re.sub(r"\s+", " ", query_text.strip().strip("?").strip(".")).lower()
        for pattern in [
            r"^what\s+is\s+(?:the\s+)?(.+)$",
            r"^define\s+(?:the\s+)?(.+)$",
            r"^explain\s+what\s+(?:the\s+)?(.+?)\s+is$",
            r"^purpose\s+of\s+(?:the\s+)?(.+)$",
            r"^what\s+is\s+shown\s+in\s+(?:the\s+)?(.+)$",
            r"^what\s+does\s+(?:the\s+)?(.+?)\s+display$",
            r"^what\s+does\s+(?:the\s+)?(.+?)\s+contain$",
        ]:
            match = re.match(pattern, q)
            if match:
                return match.group(1).strip()
        return ""

    def sentence_priority_adjustment(self, candidate: Dict[str, Any], query_text: str) -> float:
        """
        Lightweight answer-layer ranking only. Does not affect retrieval scores.
        """
        text = candidate["text"].strip()
        lowered = text.lower()
        adjustment = 0.0
        entity = self.extract_query_entity(query_text)

        action_start = lowered.startswith((
            "click ", "select ", "choose ", "press ", "open ", "navigate ",
            "after ", "then ", "enter ", "upload ", "save ", "verify ", "sign ",
        ))
        definition_pattern = bool(re.search(
            r"\b[\w\s-]{2,80}\s+(?:is|refers to|allows|is designed|contains|displays|provides)\b",
            lowered,
        ))
        attribute_pattern = bool(re.search(
            r"\b(?:contains|displays|shows|indicated|following|includes|creation date|status|file name|period|duration)\b",
            lowered,
        ))
        audit_pattern = bool(re.search(
            r"\b(?:no records|account\s*=\s*0|absent|no discrepancies|discrepancies found|fully matches|audit result)\b",
            lowered,
        ))
        procedural_pattern = lowered.startswith((
            "enter ", "click ", "select ", "choose ", "upload ", "save ", "verify ", "sign ",
        ))

        if self.is_definition_query(query_text):
            if entity and entity in lowered:
                adjustment += 0.18
            if definition_pattern:
                adjustment += 0.22
            if action_start:
                adjustment -= 0.24

        if self.is_procedural_intent_query(query_text):
            if procedural_pattern:
                adjustment += 0.18
            elif lowered.startswith(("the ", "this ", "main ", "overview ")):
                adjustment -= 0.08

        if self.is_attribute_query(query_text):
            if entity and entity in lowered:
                adjustment += 0.22
            if attribute_pattern:
                adjustment += 0.24
            if action_start and not attribute_pattern:
                adjustment -= 0.18
            if re.search(r":\s*(?:creation|status|file name|period|duration)\s*$", lowered):
                adjustment -= 0.35

        if self.is_audit_outcome_query(query_text):
            if audit_pattern:
                adjustment += 0.30
            if "generate file" in lowered or lowered.startswith(("click ", "select ")):
                adjustment -= 0.20

        return adjustment

    def has_direct_evidence(self, candidate: Dict[str, Any], query_keywords: List[str]) -> bool:
        """
        Requires sentence-level lexical support from content-bearing query terms.
        Semantic similarity alone is not enough for extractive answers in this MVP.
        """
        if not query_keywords:
            return candidate.get("score", 0.0) >= 0.45 and candidate.get("semantic_score", 0.0) >= 0.55

        raw_sentence_tokens = set(t.lower() for t in self.tokenizer.clean_and_split(candidate["text"]))
        sentence_tokens = set(self.get_query_keywords(self.tokenizer.clean_and_split(candidate["text"])))
        sentence_tokens.update(raw_sentence_tokens.intersection({"system", "information"}))
        matches = [kw for kw in query_keywords if kw in sentence_tokens]
        if query_keywords and len(matches) < len(query_keywords):
            chunk_tokens = set(self.get_query_keywords(self.tokenizer.clean_and_split(candidate.get("chunk_content", ""))))
            chunk_matches = [kw for kw in query_keywords if kw in chunk_tokens]
            action_terms = {"edit", "access", "use", "uses", "sign", "save", "import", "enter", "password"}
            if len(chunk_matches) == len(query_keywords) and any(term in sentence_tokens for term in action_terms):
                matches = chunk_matches
        if not matches:
            return False
        for required_term in {"button", "password", "email"}:
            if required_term in query_keywords and required_term not in sentence_tokens:
                return False

        coverage = len(matches) / max(len(query_keywords), 1)
        if len(query_keywords) == 1:
            return candidate.get("score", 0.0) >= 0.24
        if len(query_keywords) == 2:
            return coverage >= 1.0 and candidate.get("score", 0.0) >= 0.28
        return coverage >= 0.34 and candidate.get("score", 0.0) >= 0.28

    def evidence_is_sufficient(self, selected_candidates: List[Dict[str, Any]], retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Final confidence gate before formatting. Prevents unrelated passages from
        becoming answers when retrieval or sentence evidence is weak.
        """
        if not selected_candidates:
            return False, "No selected sentences passed direct evidence validation"

        chunk_scores = sorted([float(c.get("score", 0.0)) for c in retrieved_chunks], reverse=True)
        top_retrieval_score = chunk_scores[0] if chunk_scores else 0.0
        rank_gap = (chunk_scores[0] - chunk_scores[1]) if len(chunk_scores) > 1 else top_retrieval_score
        best_sentence_score = max(float(c.get("score", 0.0)) for c in selected_candidates)
        best_lexical = max(float(c.get("lexical_score", 0.0)) for c in selected_candidates)
        supporting_chunks = len({(c.get("document_id"), c.get("chunk_index")) for c in selected_candidates})

        if top_retrieval_score < 0.58:
            return False, f"Top retrieval score too low: {top_retrieval_score:.4f}"
        if best_lexical <= 0.0:
            return False, "No lexical overlap in selected evidence"
        if best_sentence_score < 0.26:
            return False, f"Best sentence score too low: {best_sentence_score:.4f}"
        if supporting_chunks == 0:
            return False, "No supporting chunks"

        print("\nAnswer confidence validation:")
        print(f"  Top retrieval score: {top_retrieval_score:.4f}")
        print(f"  Rank-1/Rank-2 gap: {rank_gap:.4f}")
        print(f"  Best sentence score: {best_sentence_score:.4f}")
        print(f"  Best lexical overlap: {best_lexical:.4f}")
        print(f"  Supporting chunks: {supporting_chunks}")
        return True, "Evidence passed confidence validation"

    def clean_sentence_flow(self, text: str) -> str:
        """
        Styles sentence flow by capitalizing initial letters, standardizing terminal punctuation,
        stripping leading markdown markers, and stripping leading transition words.
        """
        if not text:
            return ""
            
        # 1. Strip redundant markdown markers (bullets, headers, hashes, brackets)
        cleaned = re.sub(r'^(?:\d+[\.\)]|[\-\*•#]|\?)\s*', '', text).strip()
        repeated_heading = re.match(r'^([A-Z][A-Za-z0-9-]{2,30})\s+The\s+\1\b\s*(.*)$', cleaned)
        if repeated_heading:
            cleaned = f"The {repeated_heading.group(1)} {repeated_heading.group(2)}".strip()
        repeated_phrase_heading = re.match(
            r'^(User Profile|Data Audit|System Information|Info Panel|Headers)\s+The\s+\1\b\s*(.*)$',
            cleaned,
            flags=re.IGNORECASE,
        )
        if repeated_phrase_heading:
            phrase = repeated_phrase_heading.group(1)
            cleaned = f"The {phrase} {repeated_phrase_heading.group(2)}".strip()
        cleaned = re.sub(r'\ba Audit section\b', 'Data Audit section', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^sed to display\b', 'Used to display', cleaned, flags=re.IGNORECASE)
        
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

    def detect_contradictions(self, candidates: List[Dict[str, Any]], query_keywords: List[str] = None) -> Optional[str]:
        """
        Scans retrieved candidates for contradictory statements.
        Identifies pairs that share high keyword overlap (same topic) but mismatch
        in negation markers or modal permissions (e.g. can vs cannot, admin only vs all users).
        Returns a markdown warning block if a conflict is found.
        """
        negations = {"not", "cannot", "never", "no", "restricted", "denied", "unable", "disabled"}
        query_keywords = query_keywords or []
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                s1 = candidates[i]
                s2 = candidates[j]
                if s1.get("chunk_index") == s2.get("chunk_index"):
                    continue
                if s1.get("chunk_score", 0.0) < 0.68 or s2.get("chunk_score", 0.0) < 0.68:
                    continue
                
                tokens1 = set(w.lower() for w in re.findall(r'\w+', s1["text"]) if len(w) > 4)
                tokens2 = set(w.lower() for w in re.findall(r'\w+', s2["text"]) if len(w) > 4)
                common = tokens1.intersection(tokens2)
                union = tokens1.union(tokens2)
                topic_overlap = len(common) / len(union) if union else 0.0
                query_supported = any(kw in tokens1 and kw in tokens2 for kw in query_keywords)
                if len(common) < 3 or topic_overlap < 0.28 or not query_supported:
                    continue

                text1_lower = s1["text"].lower()
                text2_lower = s2["text"].lower()
                has_neg1 = any(re.search(rf'\b{re.escape(n)}\b', text1_lower) for n in negations)
                has_neg2 = any(re.search(rf'\b{re.escape(n)}\b', text2_lower) for n in negations)
                if has_neg1 == has_neg2:
                    continue

                ref1 = f"[{s1.get('citation_index', 1)}] {s1['filename']} (Chunk #{s1['chunk_index']})"
                ref2 = f"[{s2.get('citation_index', 2)}] {s2['filename']} (Chunk #{s2['chunk_index']})"
                warning = (
                    "> [!WARNING]\n"
                    "> **Documentation Conflict Detected:** We found potentially contradictory statements in the manual regarding this topic:\n"
                    f"> - According to **{ref1}**: *\"{s1['text']}\"*\n"
                    f"> - According to **{ref2}**: *\"{s2['text']}\"*\n"
                    "> - **Conflict Category:** opposite availability or restriction statements.\n\n"
                )
                return warning
        return None

    def generate_title(self, query_text: str) -> str:
        """
        Derives a clean, capitalized title from the user query by stripping
        common question templates and cleaning spaces.
        """
        q = " ".join(query_text.strip("?").strip(".").split())
        q_lower = q.lower()
        if q_lower == "how do i sign the document" or q_lower == "how to sign the document":
            return "# Document Signing\n\n"
        if q_lower.startswith("what information is displayed before "):
            rest = q[len("what information is displayed before "):]
            words = ["Information", "Displayed", "Before"] + re.findall(r"[A-Za-z0-9-]+", rest)
            return f"# {' '.join(words[:6]).title()}\n\n"

        for prefix in [
            "how do i ", "how to ", "what is ", "what are ", "why does ", "why do ", 
            "define ", "list of ", "list ", "explain what ", "explain ", "purpose of ",
            "can i ", "can we ", "should we ", "should i ", "does the ", "is it "
        ]:
            if q_lower.startswith(prefix):
                q = q[len(prefix):]
                break

        words = [
            w for w in re.findall(r"[A-Za-z0-9-]+", q)
            if w.lower() not in {"the", "a", "an", "it", "this", "that", "those", "them", "me", "i"}
        ]
        deduped = []
        for word in words:
            if not deduped or deduped[-1].lower() != word.lower():
                deduped.append(word)

        title = " ".join(deduped[:6]).strip().title()
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
        
        query_keywords = self.get_query_keywords(query_tokens)
        if "system information" in q_lower:
            for term in ["system", "information"]:
                if term not in query_keywords:
                    query_keywords.append(term)
        
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
                    "chunk_score": chunk_score,
                    "chunk_content": chunk.get("content", "")
                })
                
        actual_fallback_message = self.weak_evidence_message

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
            s_tokens = self.get_query_keywords(self.tokenizer.clean_and_split(c["text"]))
            if "system information" in q_lower:
                raw_tokens = set(t.lower() for t in self.tokenizer.clean_and_split(c["text"]))
                for term in ["system", "information"]:
                    if term in raw_tokens and term not in s_tokens:
                        s_tokens.append(term)
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
            base_score = float(combined_scores[i])
            priority_adjustment = self.sentence_priority_adjustment(c, query_text)
            c["score"] = max(0.0, min(1.0, base_score + priority_adjustment))
            c["vector"] = sentence_vectors[i]
            c["semantic_score"] = float(semantic_scores[i])
            c["lexical_score"] = float(norm_lexical_scores[i])
            c["priority_adjustment"] = float(priority_adjustment)

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
            print(f"    Scores -> Semantic: {c['semantic_score']:.4f} | Lexical: {c['lexical_score']:.4f} | Priority Adj: {c['priority_adjustment']:.4f} | Combined: {c['score']:.4f}")
            
        print("\n3. Final ranked sentence list (before validation):")
        ranked_for_log = sorted(candidate_sentences, key=lambda x: x["score"], reverse=True)
        for idx, r in enumerate(ranked_for_log):
            print(f"  Rank {idx+1}: Combined Score: {r['score']:.4f} | Semantic Score: {r['semantic_score']:.4f} | Lexical Score: {r['lexical_score']:.4f} | Priority Adj: {r['priority_adjustment']:.4f} | \"{r['text']}\"")
        # ----------------------------------------------------

        # 3. Filter and Rank Candidate Sentences with Answer Validation
        valid_candidates = []
        validation_logs = []
        for c in candidate_sentences:
            is_valid = self.has_direct_evidence(c, query_keywords)
            if is_valid:
                c["selection_reason"] = "Passed direct evidence validation"
                valid_candidates.append(c)
                validation_logs.append((c["text"], True, "Passed validation filters"))
            else:
                reason = "Weak evidence: no direct content-keyword support or score below threshold"
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
            selected_candidates = [
                c for c in selected_candidates
                if self.is_instructional_sentence(c["text"]) or self.has_direct_evidence(c, query_keywords)
            ]

        evidence_ok, evidence_reason = self.evidence_is_sufficient(selected_candidates, retrieved_chunks)

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
        print(f"\n7. Evidence confidence gate: {'PASSED' if evidence_ok else 'FAILED'}")
        print(f"    Reason: {evidence_reason}")
        print("="*70 + "\n")
        # ----------------------------------------------------

        if not evidence_ok:
            return self.weak_evidence_message, []
            
        # 5. Classify Query Type and Route Formatting
        is_how = any(word in q_lower for word in ["how", "steps", "procedure", "instructions", "guide"])
        is_summarize = (
            q_lower.startswith(("explain", "describe", "walk me through", "summarize"))
            or "what happens after" in q_lower
            or q_lower.startswith("how does")
        )
        is_what = any(word in q_lower for word in ["what is", "define", "who is"])
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
        contradiction_warning = self.detect_contradictions(selected_candidates, query_keywords)
            
        # Format: explanatory summary -> concise multi-sentence answer
        if is_summarize:
            selected_candidates = selected_candidates[:4]
            selected_candidates.sort(key=lambda x: (x["chunk_index"], x["sentence_index"]))
            parts = []
            for s in selected_candidates:
                clean_text = self.clean_sentence_flow(s["text"])
                if clean_text:
                    parts.append(f"{clean_text} [{s['citation_index']}]")
            answer_body = " ".join(parts)

        # Format: "Compare" -> comparison table
        elif is_compare:
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
            def_triggers = [" is a", " is designed for", " refers to", " is defined as", " means", " represents"]
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
            instructional_candidates = [s for s in selected_candidates if self.is_instructional_sentence(s["text"])]
            if instructional_candidates:
                selected_candidates = instructional_candidates
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
