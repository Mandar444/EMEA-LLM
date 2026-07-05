import re
from typing import Dict, List, Tuple, Set, Optional

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the edit distance (Levenshtein distance) between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

class QueryUnderstandingLayer:
    """
    Lightweight Query Preprocessing & Understanding Layer.
    Detects question intent, handles spelling mistakes, expands abbreviations,
    and maps synonyms using vocabulary from our custom corpus.
    """
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self.vocab = vocab_dict or {}
        
        # 1. Abbreviations Mapping
        self.abbreviations = {
            "gl": "general ledger",
            "saf-t": "standard audit file for tax",
            "saft": "standard audit file for tax",
            "auth": "authorization",
            "pwd": "password",
            "pass": "password",
            "admin": "administrator",
            "db": "database",
            "doc": "document",
            "docs": "documents",
            "ui": "interface",
            "nav": "navigation",
            "reg": "registration",
            "config": "configuration",
            "setup": "settings"
        }
        
        # 2. Synonym Mappings
        self.synonyms = {
            "sign in": "log in",
            "sign-in": "log in",
            "logon": "log in",
            "log-on": "log in",
            "remove": "delete",
            "erase": "delete",
            "wipe": "delete",
            "change": "edit",
            "modify": "edit",
            "adjust": "edit",
            "who": "administrator",
            "register": "create",
            "add": "create",
            "show": "display"
        }
        
        # Stopwords to ignore in OOV spellcheck
        self.stopwords = {
            "what", "is", "how", "to", "the", "a", "an", "do", "i", "system", "of", "in", "and", "or", 
            "for", "with", "on", "at", "by", "from", "about", "this", "that", "these", "those", "who", 
            "whom", "whose", "which", "where", "when", "why", "can", "should", "could"
        }

    def correct_spelling(self, tokens: List[str]) -> List[str]:
        """
        Corrects spelling mistakes using edit distance against the custom corpus vocabulary.
        Only corrects words not in vocabulary, length > 3, and not stopwords or numbers.
        """
        if not self.vocab:
            return tokens
            
        corrected = []
        for token in tokens:
            # Skip short words, numbers, punctuation, or words already in vocab
            if (len(token) <= 3 or 
                token in self.vocab or 
                token in self.stopwords or 
                re.match(r'^\d+$', token) or 
                re.match(r'^[^\w\s]$', token)):
                corrected.append(token)
                continue
                
            # Find closest vocabulary word
            best_word = token
            min_dist = 999
            
            for vocab_word in self.vocab.keys():
                # Optimization: skip words with large length differences
                if abs(len(vocab_word) - len(token)) > 2:
                    continue
                # Skip helper tokens
                if vocab_word.startswith("<") and vocab_word.endswith(">"):
                    continue
                    
                dist = levenshtein_distance(token, vocab_word)
                if dist < min_dist:
                    min_dist = dist
                    best_word = vocab_word
            
            # Apply correction if edit distance is small:
            # - For short tokens (<= 5 chars), require edit distance 1 to prevent aggressive OOV mapping (e.g. login -> logo)
            # - For longer tokens, allow edit distance <= 2
            max_allowed = 1 if len(token) <= 5 else 2
            if min_dist <= max_allowed:
                corrected.append(best_word)
            else:
                corrected.append(token)
                
        return corrected

    def expand_abbreviations_and_synonyms(self, query_text: str) -> str:
        """
        Normalizes abbreviations and maps synonyms in query text.
        """
        text = query_text.lower().strip()
        
        # 1. Expand multi-word synonyms first (e.g. "sign in" -> "log in")
        for syn_key, syn_val in self.synonyms.items():
            if " " in syn_key:
                text = re.sub(r'\b' + re.escape(syn_key) + r'\b', syn_val, text)
                
        # 2. Tokenize and expand single-word abbreviations/synonyms
        words = re.findall(r'\w+|[^\w\s]', text)
        processed_words = []
        
        for w in words:
            if w in self.abbreviations:
                processed_words.append(self.abbreviations[w])
            elif w in self.synonyms:
                processed_words.append(self.synonyms[w])
            else:
                processed_words.append(w)
                
        return " ".join(processed_words)

    def classify_intent(self, query_text: str) -> str:
        """
        Classifies the intent / format type of query:
        procedural, definition, explanation, navigation, comparison, list, yes_no, or default.
        """
        q = query_text.lower()
        
        # Check comparison
        if any(word in q for word in ["compare", "difference", "versus", "vs", "difference between"]):
            return "comparison"
            
        # Check navigation
        if any(word in q for word in ["where is", "where do i find", "location of", "navigating to"]):
            return "navigation"
            
        # Check procedural
        if any(word in q for word in ["how do i", "how to", "steps for", "procedure to", "instructions for"]):
            return "procedural"
            
        # Check lists
        if any(word in q for word in ["list of", "list features", "items", "categories"]):
            return "list"
            
        # Check yes/no/confirmations
        if any(q.startswith(word) for word in [
            "can ", "should ", "could ", "is it ", "does the ", "do we ", "is there ", "are there "
        ]):
            return "yes_no"
            
        # Check definition
        if any(word in q for word in ["what is", "define", "definition of", "what does"]):
            return "definition"
            
        # Check explanation
        if any(word in q for word in ["why do", "why does", "reason for"]):
            return "explanation"
            
        return "default"

    def preprocess_query(self, query_text: str) -> Tuple[str, List[str], str]:
        """
        Main query understanding gateway:
        1. Classifies query intent
        2. Normalizes, expands abbreviations and maps synonyms
        3. Corrects spellings of out-of-vocabulary words
        Returns (rewritten_query_string, query_token_list, intent)
        """
        intent = self.classify_intent(query_text)
        
        # Apply expansions
        expanded = self.expand_abbreviations_and_synonyms(query_text)
        
        # Tokenize expanded query
        tokens = re.findall(r'\w+|[^\w\s]', expanded)
        
        # Apply spelling corrections
        corrected_tokens = self.correct_spelling(tokens)
        
        # Cleaned tokens list (removing punctuation/spaces for index overlap checks)
        clean_tokens = [t for t in corrected_tokens if re.match(r'^\w+$', t)]
        
        rewritten_query = " ".join(corrected_tokens)
        return rewritten_query, clean_tokens, intent
