import sys
from collections import defaultdict, Counter

def get_vocabulary(corpus):
    """
    make initial vocabulary from corpus
    """
    # make defaultdict, so that we can get 0 for words we haven't seen
    vocab = defaultdict(int)
    
    # for every line
    for line in corpus:
        # get each word
        words = line.strip().split()
        
        for word in words:
            # make word into space separated characters
            # like 'cat' becomes "c a t </w>"
            # The </w> is our special end-of-word marker
            word_chars = ' '.join(list(word)) + ' </w>'
            
            # Count how many times we see this character sequence
            vocab[word_chars] += 1
    
    return vocab

def get_stats(vocab):
    """
    count frequency of adjacent pairs
    """
    # Counter is like a dictionary optimized for counting things
    pairs = Counter()
    
    # for each word in our vocabulary
    for word, frequency in vocab.items():
        # split word into individual symbols
        symbols = word.split()
        
        # for each adjacent pair of symbols
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            # add frequency of this word to the count for this pair
            pairs[pair] += frequency
    
    return pairs

def merge_vocab(pair, vocab):
    """
    merge the most frequent pair in the vocabulary
    """
    new_vocab = {}
    
    # the string pattern we want 
    # For pair ("e", "r"), bigram is "e r" (with a space)
    bigram = ' '.join(pair)
    
    # the replacement we need
    # For pair ("e", "r"), replacement is "er"
    replacement = ''.join(pair)
    
    # Go through each word in our vocabulary
    for word in vocab:
        # new word after replacement
        new_word = word.replace(bigram, replacement)
        # update vocab
        new_vocab[new_word] = vocab[word]
    
    return new_vocab

def byte_pair_encoding(corpus_file, num_merges):
    """
    BPE algorithm:
    1. Start with each character as a separate token
    2. Find the most frequent pair of adjacent tokens
    3. Merge that pair into a single token
    4. Repeat steps 2-3 for k iterations
    """
    # read corpus from file
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.readlines()
    
    # Initialize our vocabulary with individual characters
    vocab = get_vocabulary(corpus)
    
    print(f"Initial vocab size: {len(vocab)}")
    
    # perform k merges as asked
    for i in range(num_merges):
        pairs = get_stats(vocab)
        
        if not pairs:
            print(f"Stopping at {i} merges")
            break
        # get the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        
        print(f"itteration {i+1}: Merging pair {best_pair} (frequency: {pairs[best_pair]})")
        
        # Make new vocab
        vocab = merge_vocab(best_pair, vocab)
    
    print(f"Final vocab size: {len(vocab)}")
    
    return vocab

def extract_tokens(vocab):
    """
    Get all unique tokens from the vocab

    if vocab contains "pl ay er </w>" and "pl ay </w>",
    we extract the tokens: pl, ay, er, </w>
    """
    # set for handling duplicates
    tokens = set()
    
    # for each word in vocab
    for word in vocab.keys():
        # split word into tokens
        word_tokens = word.split()
        
        # add each token to set
        for token in word_tokens:
            tokens.add(token)
    
    # return as sorted list for consistent output
    return sorted(tokens)


if __name__ == "__main__":
    # Check if the user provided the right number of arguments
    if len(sys.argv) < 3:
        print("Usage: python B23CS1037_prob2.py k corpus.txt")
        sys.exit(1)
    
    # get number of merges
    num_merges = int(sys.argv[1])
    
    # get file path
    corpus_file = sys.argv[2]
    
    # run BPE
    final_vocab = byte_pair_encoding(corpus_file, num_merges)
    
    # get all token
    tokens = extract_tokens(final_vocab)
    
    # show result
    print("all vocab tokens:")
    
    for token in tokens:
        print(token)

    print(f"Total tokens: {len(tokens)}")
