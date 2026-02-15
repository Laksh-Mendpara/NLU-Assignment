import math
from collections import defaultdict

class NaiveBayesClassifier:
    """
    Naive Bayes classifier for sentiment analysis.
    """
    
    def __init__(self):
        # word counts for each class
        self.word_counts = {'positive': defaultdict(int), 'negative': defaultdict(int)}
        
        # total number of words in each class
        self.class_word_totals = {'positive': 0, 'negative': 0}
        
        # number of documents (sentences) belong to each class
        self.class_doc_counts = {'positive': 0, 'negative': 0}
        
        # vocabulary
        self.vocabulary = set()
        
        # total number of documents we've trained on
        self.total_docs = 0
    
    def tokenize(self, text):
        """
        Tokenize the input text.
        currently splits on whitespace.
        """
        return text.lower().split()
    
    def train(self, positive_sentences, negative_sentences):
        """
        Train the classifier on the given positive and negative sentences.
        """
        
        # process all the positive sentences
        for sentence in positive_sentences:
            sentence = sentence.strip()
            if sentence == "":
                continue
            # Count this as one more positive document
            self.class_doc_counts['positive'] += 1
            
            # Break sentence into words
            tokens = self.tokenize(sentence)
            
            # Count each word
            for word in tokens:
                # Increment the count for this word in positive class
                self.word_counts['positive'][word] += 1
                
                # Also count towards the total positive words
                self.class_word_totals['positive'] += 1
                
                # Add this word to our overall vocabulary
                self.vocabulary.add(word)
        
        # for negative sentences
        for sentence in negative_sentences:
            sentence = sentence.strip()
            if sentence == "":
                continue
            self.class_doc_counts['negative'] += 1
            tokens = self.tokenize(sentence)
                
            for word in tokens:
                self.word_counts['negative'][word] += 1
                self.class_word_totals['negative'] += 1
                self.vocabulary.add(word)
        
        # Calculate total number of documents we trained on
        self.total_docs = self.class_doc_counts['positive'] + self.class_doc_counts['negative']
        
        # Show the user what we learned
        print(f"Training complete!")
        print(f"Positive documents: {self.class_doc_counts['positive']}")
        print(f"Negative documents: {self.class_doc_counts['negative']}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def predict(self, sentence):
        """
        predict the sentiment of a sentence
        """
        # Tokenize the input sentence
        tokens = self.tokenize(sentence)
        
        # Calculate prior probabilities
        # P(positive) = number of positive docs / total docs
        # We use log() to get log probabilities
        log_prior_pos = math.log(self.class_doc_counts['positive'] / self.total_docs)
        log_prior_neg = math.log(self.class_doc_counts['negative'] / self.total_docs)
        
        # Get vocabulary size for Laplace smoothing
        vocab_size = len(self.vocabulary)
        
        # Calculate likelihood for positive class
        # Start with the prior probability
        log_likelihood_pos = 0
        
        for word in tokens:
            # How many times did this word appear in positive documents?
            word_count = self.word_counts['positive'][word]
            
            # Apply Laplace smoothing (add-one smoothing):
            # We add 1 to the numerator and vocab_size to the denominator
            # This prevents zero probabilities for words we haven't seen
            # Formula: P(word|positive) = (count + 1) / (total_positive_words + vocab_size)
            probability = (word_count + 1) / (self.class_word_totals['positive'] + vocab_size)
            
            # Add the log probability (log of product = sum of logs)
            log_likelihood_pos += math.log(probability)
        
        # calculate likelihood for negative class
        log_likelihood_neg = 0
        
        for word in tokens:
            word_count = self.word_counts['negative'][word]
            # Same Laplace smoothing formula
            probability = (word_count + 1) / (self.class_word_totals['negative'] + vocab_size)
            log_likelihood_neg += math.log(probability)
        
        # Calculate posterior probabilities by combining prior and likelihood
        # P(positive|sentence) ∝ P(positive) × P(sentence|positive)
        log_posterior_pos = log_prior_pos + log_likelihood_pos
        log_posterior_neg = log_prior_neg + log_likelihood_neg
        
        # Choose the class with higher posterior probability
        if log_posterior_pos > log_posterior_neg:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

def load_data(pos_file, neg_file):
    """
    load data
    """
    # open positive sentences file
    with open(pos_file, 'r', encoding='utf-8') as f:
        positive_sentences = f.readlines()
    
    # open negative sentences file
    with open(neg_file, 'r', encoding='utf-8') as f:
        negative_sentences = f.readlines()
    
    return positive_sentences, negative_sentences

def main():
    """
    main function
    """
    
    # data path
    pos_file = 'data/pos.txt'
    neg_file = 'data/neg.txt'
    
    print(f"data loaded from {pos_file} and {neg_file}")
    positive_sentences, negative_sentences = load_data(pos_file, neg_file)
    
    if positive_sentences is None or negative_sentences is None:
        print("Error: could not load data, check file paths")
        return
    
    # make classifier instance 
    classifier = NaiveBayesClassifier()
    
    # train classifier
    classifier.train(positive_sentences, negative_sentences)
    
    # inf. loop
    print(f"Enter 'quit' or 'exit' to stop.")
    while True:
        # Get user input
        sentence = input("Enter sentence: ").strip()
        
        # Check if user wants to exit
        if sentence.lower() in ['quit', 'exit']:
            print("bye!")
            break
        
        # predict
        sentiment = classifier.predict(sentence)
        print(f"Prediction: {sentiment}\n")

if __name__ == "__main__":
    main()
