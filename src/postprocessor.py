"""
TinyWorld OCR - Post-processing Module
Spell correction and text cleanup using dictionary and heuristics
"""

import re
from collections import Counter


class PostProcessor:
    """Post-processing for OCR text correction"""
    
    def __init__(self, dictionary_path='dictionaries/common_words.txt'):
        """
        Initialize post-processor
        
        Args:
            dictionary_path: Path to dictionary file
        """
        self.dictionary = set()
        self.word_freq = Counter()
        self._load_dictionary(dictionary_path)
        
        # Common OCR substitution errors
        self.substitutions = {
            '0': 'O', 'O': '0',
            '1': 'l', 'l': '1', 'I': '1',
            '5': 'S', 'S': '5',
            '8': 'B', 'B': '8',
            'rn': 'm', 'vv': 'w',
        }
    
    def _load_dictionary(self, dictionary_path):
        """Load dictionary from file"""
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.dictionary.add(word)
                        # Assume frequency decreases with position
                        self.word_freq[word] = len(self.dictionary) - len(self.word_freq)
            print(f"Loaded {len(self.dictionary)} words from dictionary")
        except FileNotFoundError:
            print(f"Warning: Dictionary not found at {dictionary_path}")
            print("Running without spell correction")
    
    def edit_distance(self, s1, s2):
        """
        Calculate Levenshtein edit distance
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self.edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertion, deletion, substitution
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_candidates(self, word, max_distance=2):
        """
        Get candidate corrections for a word
        
        Args:
            word: Word to correct
            max_distance: Maximum edit distance
            
        Returns:
            List of (candidate, distance) tuples
        """
        word_lower = word.lower()
        
        # If word is in dictionary, return it
        if word_lower in self.dictionary:
            return [(word, 0)]
        
        # Find candidates within edit distance
        candidates = []
        for dict_word in self.dictionary:
            # Quick length filter
            if abs(len(dict_word) - len(word_lower)) > max_distance:
                continue
            
            dist = self.edit_distance(word_lower, dict_word)
            if dist <= max_distance:
                candidates.append((dict_word, dist))
        
        # Sort by distance, then by frequency
        candidates.sort(key=lambda x: (x[1], -self.word_freq.get(x[0], 0)))
        
        return candidates[:5]  # Top 5 candidates
    
    def correct_word(self, word, context_words=None):
        """
        Correct a single word
        
        Args:
            word: Word to correct
            context_words: Surrounding words for context
            
        Returns:
            Corrected word
        """
        # Skip very short words and numbers
        if len(word) <= 1 or word.isdigit():
            return word
        
        # Check common substitutions
        for wrong, right in self.substitutions.items():
            if wrong in word:
                test_word = word.replace(wrong, right)
                if test_word.lower() in self.dictionary:
                    return test_word
        
        # Get candidates
        candidates = self.get_candidates(word, max_distance=2)
        
        if not candidates:
            return word  # No correction found
        
        # Return best candidate
        best_candidate, distance = candidates[0]
        
        # Only correct if confident
        if distance <= 1:
            # Preserve original casing
            if word.isupper():
                return best_candidate.upper()
            elif word[0].isupper():
                return best_candidate.capitalize()
            else:
                return best_candidate
        
        return word  # Keep original if not confident
    
    def correct_text(self, text, aggressive=False):
        """
        Correct entire text
        
        Args:
            text: Input text
            aggressive: Whether to use aggressive correction
            
        Returns:
            Corrected text
        """
        if not self.dictionary:
            return text  # No dictionary, return as is
        
        # Split into words while preserving punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        corrected_words = []
        for i, word in enumerate(words):
            # Skip punctuation
            if not word.isalnum():
                corrected_words.append(word)
                continue
            
            # Get context
            context = None
            if i > 0:
                context = words[i-1:i+2]
            
            # Correct word
            corrected = self.correct_word(word, context)
            corrected_words.append(corrected)
        
        # Reconstruct text
        result = ""
        for i, word in enumerate(corrected_words):
            result += word
            # Add space after word (but not after punctuation at end)
            if i < len(corrected_words) - 1 and corrected_words[i+1] not in '.,!?;:':
                if word not in '.,!?;:(' and corrected_words[i+1] != ')':
                    result += " "
        
        return result
    
    def clean_text(self, text):
        """
        Clean up text formatting
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        
        # Fix spacing around quotes
        text = re.sub(r'\s+"', ' "', text)
        text = re.sub(r'"\s+', '" ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def post_process(self, text, use_spell_correction=True, aggressive=False):
        """
        Complete post-processing pipeline
        
        Args:
            text: Input OCR text
            use_spell_correction: Whether to apply spell correction
            aggressive: Use aggressive correction
            
        Returns:
            Post-processed text
        """
        # Clean formatting
        text = self.clean_text(text)
        
        # Spell correction
        if use_spell_correction and self.dictionary:
            text = self.correct_text(text, aggressive=aggressive)
        
        # Final cleanup
        text = self.clean_text(text)
        
        return text


def create_sample_dictionary(output_path='dictionaries/common_words.txt', 
                            num_words=10000):
    """
    Create a sample dictionary file with most common English words
    
    Args:
        output_path: Path to save dictionary
        num_words: Number of words to include
    """
    import os
    
    # Common English words (most frequent)
    common_words = [
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
        'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
        'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
        'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
        'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
        # Add more words...
        'text', 'image', 'document', 'page', 'line', 'word', 'character', 'file',
        'system', 'process', 'data', 'information', 'result', 'method', 'example',
        'number', 'value', 'type', 'form', 'part', 'place', 'case', 'point', 'hand',
        'important', 'different', 'small', 'large', 'great', 'high', 'old', 'long',
        'right', 'left', 'top', 'bottom', 'center', 'side', 'end', 'begin', 'start',
    ]
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in common_words[:num_words]:
            f.write(word + '\n')
    
    print(f"Created dictionary with {len(common_words[:num_words])} words at {output_path}")


if __name__ == '__main__':
    # Create sample dictionary
    create_sample_dictionary()
    
    # Test post-processor
    processor = PostProcessor()
    
    # Test text with errors
    test_text = "Th1s  is  a  t3st  0f  th3  p0st-pr0cessor. It  sh0uld  c0rrect  c0mm0n  err0rs."
    print("\nOriginal text:")
    print(test_text)
    
    corrected = processor.post_process(test_text, use_spell_correction=True)
    print("\nCorrected text:")
    print(corrected)