"""
Tokenization Class Architecture

```
Tokenization System Structure:
┌─────────────────────────────────┐
│ Base Tokenizer Interface:       │
│ • encode(text) → token_ids      │
│ • decode(token_ids) → text      │
├─────────────────────────────────┤
│ CharTokenizer (Simple):         │
│ • vocab: list of characters     │
│ • char_to_id: lookup mapping    │
│ • id_to_char: reverse mapping   │
├─────────────────────────────────┤
│ BPETokenizer (Advanced):        │
│ • vocab: learned subwords       │
│ • merges: learned pair rules    │
│ • token_to_id/id_to_token maps  │
├─────────────────────────────────┤
│ Utility Functions:              │
│ • create_tokenizer()            │
│ • tokenize_dataset()            │
│ • analyze_tokenization()        │
└─────────────────────────────────┘
```


-----------

Byte Pair Encoding (BPE) Tokenizer

BPE is the secret sauce behind modern language models (GPT, BERT, etc.). It learns to merge frequent character pairs,
creating subword units that balance vocabulary size with sequence length.

```
┌───────────────────────────────────────────────────────────────────────┐
│ BPE TRAINING ALGORITHM: Learning Subword Units                        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ STEP 1: Initialize with Character Vocabulary                          │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Training Data: ["hello", "hello", "help"]                         │ │
│ │                                                                   │ │
│ │ Initial Tokens (with end-of-word markers):                        │ │
│ │   ['h','e','l','l','o</w>']    (hello)                            │ │
│ │   ['h','e','l','l','o</w>']    (hello)                            │ │
│ │   ['h','e','l','p</w>']        (help)                             │ │
│ │                                                                   │ │
│ │ Starting Vocab: ['h', 'e', 'l', 'o', 'p', '</w>']                 │ │
│ │                   ↑ All unique characters                         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 2: Count All Adjacent Pairs                                      │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Pair Frequency Analysis:                                          │ │
│ │                                                                   │ │
│ │   ('h', 'e'): ██████  3 occurrences  ← MOST FREQUENT!             │ │
│ │   ('e', 'l'): ██████  3 occurrences                               │ │
│ │   ('l', 'l'): ████    2 occurrences                               │ │
│ │   ('l', 'o'): ████    2 occurrences                               │ │
│ │   ('o', '<'): ████    2 occurrences                               │ │
│ │   ('l', 'p'): ██      1 occurrence                                │ │
│ │   ('p', '<'): ██      1 occurrence                                │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 3: Merge Most Frequent Pair                                      │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Merge Operation: ('h', 'e') → 'he'                                │ │
│ │                                                                   │ │
│ │ BEFORE:                          AFTER:                           │ │
│ │   ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']            │ │
│ │   ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']            │ │
│ │   ['h','e','l','p</w>']      →  ['he','l','p</w>']                │ │
│ │                                                                   │ │
│ │ Updated Vocab: ['h','e','l','o','p','</w>', 'he']                 │ │
│ │                                              ↑ NEW TOKEN!         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 4: Repeat Until Target Vocab Size Reached                        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Iteration 2: Next most frequent is ('l', 'l')                     │ │
│ │ Merge ('l','l') → 'll'                                            │ │
│ │                                                                   │ │
│ │   ['he','l','l','o</w>']     →  ['he','ll','o</w>']               │ │
│ │   ['he','l','l','o</w>']     →  ['he','ll','o</w>']               │ │
│ │   ['he','l','p</w>']         →  ['he','l','p</w>']                │ │
│ │                                                                   │ │
│ │ Updated Vocab: ['h','e','l','o','p','</w>','he','ll']             │ │
│ │                                                  ↑ NEW!           │ │
│ │                                                                   │ │
│ │ Continue merging until vocab_size target...                       │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ FINAL RESULTS:                                                        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Trained BPE can now encode efficiently:                           │ │
│ │                                                                   │ │
│ │ "hello" → ['he', 'll', 'o</w>']  = 3 tokens (vs 5 chars)          │ │
│ │ "help"  → ['he', 'l', 'p</w>']   = 3 tokens (vs 4 chars)          │ │
│ │                                                                   │ │
│ │  Key Insights: BPE automatically discovers:                       │ │
│ │    - Common prefixes ('he')                                       │ │
│ │    - Morphological patterns ('ll')                                │ │
│ │    - Natural word boundaries (</w>)                               │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```
"""

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

KB_TO_BYTES = 1024


class Tokenizer:
    """
    Base tokenizer class providing the interface for all tokenizers.

    This defines the contract that all tokenizers must follow:
        - encode(): text -> list of token IDs
        - decode(): list of token IDs -> text
    """

    # Predefined symbolic tokens for common use cases
    TOK_UNKNOWN = "<UNK>"  # Unknown
    TOK_EOW = "</w>"       # END OF WORD

    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.encode("abc")
        [1, 2, 3]
        """
        raise NotImplementedError(
            "encode() not implemented in base Tokenizer class\n"
        )

    def decode(self, tokens: List[int]) -> str:
        """
        Convert list of token IDs back to text.

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.decode([1, 2, 3])
        "abc"
        """
        raise NotImplementedError(
            "decode() not implemented in base Tokenizer class\n"
        )
    

class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer that treats each character as a separate token.

    This is the simplest tokenization approach - every character in the
    vocabulary gets its own unique ID.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        """Initialize character tokenizer."""

        if vocab is None:
            vocab = []

        # Add special unknown token
        self.vocab = [Tokenizer.TOK_UNKNOWN] + vocab
        self.vocab_size = len(self.vocab)

        # Create bidirectional mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Store unknown token ID
        self.unk_id = 0

    def build_vocab(self, corpus: List[str]) -> None:
        """Build vocabulary from a corpus of text."""

        # Collect all unique characters
        all_chars = set()
        for text in corpus:
            all_chars.update(text)

        # Sort for consistent ordering
        unique_chars = sorted(all_chars)

        # Rebuild vocabulary with <UNK> token first
        self.vocab = [Tokenizer.TOK_UNKNOWN] + unique_chars
        self.vocab_size = len(self.vocab)

        # Rebuild mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """Encode text to list of character IDs."""

        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode list of token IDs back to text."""

        chars = []
        for token_id in tokens:
            # Use unknown token for invalid IDs
            char = self.id_to_char.get(token_id, Tokenizer.TOK_UNKNOWN)
            chars.append(char)
        return ''.join(chars)


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer that learns subword units.

    BPE works by:
        1. Starting with character-level vocabulary
        2. Finding most frequent character pairs
        3. Merging frequent pairs into single tokens
        4. Repeating until desired vocabulary size
    """

    def __init__(self, vocab_size: int = 1000):
        """Initialize BPE tokenizer."""
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []  # List of (pair, new_token) merges
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_word_tokens(self, word: str) -> List[str]:
        """
        Convert word to list of characters with end-of-word marker.

        EXAMPLE:
        >>> tokenizer._get_word_tokens("hello")
        ['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW]
        """
        if not word:
            return []

        tokens = list(word)
        tokens[-1] += Tokenizer.TOK_EOW
        return tokens

    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """
        Get all adjacent pairs from word tokens.

        1. Iterate through adjacent tokens
        2. Create pairs of consecutive tokens
        3. Return set of unique pairs

        EXAMPLE:
        >>> tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW])
        {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o'+Tokenizer.TOK_EOW)}
        """
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i+1]))
        return pairs

    @staticmethod
    def _count_byte_pairs(word_tokens: Dict[str, List[str]], word_freq: Counter) -> Counter:
        """
        Count frequency of all adjacent token pairs across all words.
    
        Each pair's count is weighted by how often its containing word appears
        in the corpus, so frequent words contribute more to pair statistics.
    
        ```
        Count Pairs Across All Words (weighted by frequency):
        
          word_tokens:                         word_freq:
          "hello" → ['h','e','l','l','o</w>']  freq=3
          "help"  → ['h','e','l','p</w>']      freq=1
        
          Pair counting (freq-weighted):
            ('h','e'):  3+1 = 4   ← appears in both words
            ('e','l'):  3+1 = 4   ← appears in both words
            ('l','l'):  3   = 3   ← only in "hello"
            ('l','o</w>'): 3 = 3  ← only in "hello"
            ('l','p</w>'): 1 = 1  ← only in "help"
        ```
    
        EXAMPLE:
        >>> word_tokens = {"hello": ['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW]}
        >>> word_freq = Counter({"hello": 3})
        >>> counts = _count_byte_pairs(word_tokens, word_freq)
        >>> counts[('h', 'e')]
        3
        """
        pair_counts = Counter()
    
        for word, freq in word_freq.items():
            tokens = word_tokens[word]
            # Count adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_counts[pair] += freq
        return pair_counts
    
    @staticmethod
    def _merge_pair(word_tokens: Dict[str, List[str]], pair: Tuple[str, str]) -> str:
        """
        Merge the most frequent pair in all word token lists.
    
        Scans through every word's tokens and replaces adjacent occurences
        of the pair with a single concatenated token. Modifies word_tokens
        in place and returns the new merged token string.
    
        EXAMPLE:
        >>> word_tokens = {"hello": ['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW]}
        >>> merged = _merge_pair(word_tokens, ('h', 'e'))
        >>> word_tokens["hello"]
        ['he', 'l', 'l', 'o'+Tokenizer.TOK_EOW]
        >>> merged
        'he'
        """
    
        merged_token = pair[0] + pair[1]
    
        for word in word_tokens:
            tokens = word_tokens[word]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == pair[0] and
                    tokens[i+1] == pair[1]):
                        new_tokens.append(merged_token)
                        i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            word_tokens[word] = new_tokens
    
        return merged_token

    def train(self, corpus: List[str], vocab_size: int | None = None) -> None:
        """
        Train BPE on corpus to learn merge rules.

        This is the composition function: It initializes character vocabulary,
        then runs a greedy merge loop using _count_byte_pairs() to find the
        best pair and _merge_pair() to apply it.

        APPROACH:
        1. Build initial character vocabulary from corpus words
        2. Loop: count pairs, find best, merge it, add to vocab
        3. Stop when vocab reaches target size or no pairs remain
        4. Build final mappings

        EXAMPLE:
        >>> corpus = ["hello", "hello", "help"]
        >>> tokenizer = BPETokenizer(vocab_size=20)
        >>> tokenizer.train(corpus)
        >>> len(tokenizer.vocab) <= 20
        True
        """

        if vocab_size:
            self.vocab_size = vocab_size

        # Count word frequencies and initialize character vocabulary
        word_freq = Counter(corpus)
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)

        self.vocab = sorted(vocab)
        if Tokenizer.TOK_UNKNOWN not in vocab:
            self.vocab = [Tokenizer.TOK_UNKNOWN] + self.vocab

        # Greedy merge loop: count pairs, merge best, repeat
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            pair_counts = self._count_byte_pairs(word_tokens, word_freq)
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            merged_token = self._merge_pair(word_tokens, best_pair)
            self.vocab.append(merged_token)
            self.merges.append(best_pair)

        self._build_mappings()

    def _build_mappings(self):
        """Build token-to-ID and ID-to-token mappings."""
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """
        Apply learned merge rules to token sequence.

        EXAMPLE:
        >>> # After training, merges might be [('h','e'), ('l','l')]
        >>> tokenizer._apply_merges(['h','e','l','l','o'+Tokenizer.TOK_EOW])
        ['he','ll','o'+Tokenizer.TOK_EOW]  # Applied both merges
        """

        if not self.merges:
            return tokens

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == merge_pair[0] and
                    tokens[i + 1] == merge_pair[1]):
                    # Apply merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE.

        EXAMPLE:
        >>> tokenizer.encode("hello world")
        [12, 45, 78]  # Token IDs after BPE merging
        """

        if not self.vocab:
            return []

        # Simple word splitting (could be more sophisticated)
        words = text.split()
        all_tokens = []

        for word in words:
            # Get character-level tokens
            word_tokens = self._get_word_tokens(word)

            # Apply BPE merges
            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        # Convert to IDs
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.token_to_id.get(token, 0))  # 0 = <UNK>

        return token_ids

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        EXAMPLE:
        >>> tokenizer.decode([12, 45, 78])
        "hello world"  # Reconstructed text
        """

        if not self.id_to_token:
            return ""

        # Convert IDs to tokens
        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, Tokenizer.TOK_UNKNOWN)
            token_strings.append(token)

        # Join and clean up
        text = ''.join(token_strings)

        # Replace end-of-word markers with spaces
        text = text.replace(Tokenizer.TOK_EOW, ' ')

        # Clean up extra spaces
        text = ' '.join(text.split())

        return text


def create_tokenizer(strategy: str = "char", vocab_size: int = 1000, corpus: List[str] = None) -> Tokenizer:
    """
    Factory function to create and traing tokenizer.

    >>> corpus = ["hello world", "test text"]
    >>> tokenizer = create_tokenizer("char", corpus=corpus)
    >>> tokens = tokenizer.encode("hello")
    """

    if strategy == "char":
        tokenizer = CharTokenizer()
        if corpus:
            tokenizer.build_vocab(corpus)
    elif strategy == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if corpus:
            tokenizer.train(corpus, vocab_size)
    else:
        raise ValueError(
            f"Unknown tokenization strategy: '{strategy}'\n"
        )

    return tokenizer


def tokenize_dataset(texts: List[str], tokenizer: Tokenizer, max_length: int = None) -> List[List[int]]:
    """
    Tokenize a dataset with optional length limits.

    EXAMPLE:
    >>> texts = ["hello world", "tokenize this"]
    >>> tokenizer = CharTokenizer(['h','e','l','o',' ','w','r','d','t','k','n','i','z','s'])
    >>> tokenized = tokenize_dataset(texts, tokenizer, max_length=10)
    >>> all(len(seq) <= 10 for seq in tokenized)
    True
    """
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        # Apply length limit
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return tokenized

def analyze_tokenization(texts: List[str], tokenizer: Tokenizer) -> Dict[str, float]:
    """
    Analyze tokenization statistics.

    EXAMPLE:
    >>> texts = ["hello", "world"]
    >>> tokenizer = CharTokenizer(['h','e','l','o','w','r','d'])
    >>> stats = analyze_tokenization(texts, tokenizer)
    >>> 'vocab_size' in stats and 'avg_sequence_length' in stats
    True
    """

    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    # Calculate statistics
    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]

    stats = {
        'vocab_size': tokenizer.vocab_size,
        'avg_sequence_length': np.mean(tokenized_lengths),
        'max_sequence_length': max(tokenized_lengths) if tokenized_lengths else 0,
        'total_tokens': len(all_tokens),
        'compression_ratio': total_chars / len(all_tokens) if all_tokens else 0,
        'unique_tokens': len(set(all_tokens))
    }

    return stats