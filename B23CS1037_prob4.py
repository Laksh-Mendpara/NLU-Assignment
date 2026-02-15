import re
import math
from collections import defaultdict, Counter

# sports data
SPORTS_DATA = [
    "The team won the championship after an intense final match that lasted three hours",
    "The star player scored the winning goal in the last minute of the game",
    "The coach announced new training strategies for the upcoming season",
    "The athlete broke the world record in the 100 meter sprint competition",
    "The basketball team secured their playoff position with yesterday's victory",
    "The football match ended in a dramatic penalty shootout",
    "The tennis player advanced to the semifinals after a tough match",
    "The olympic gold medalist retired from professional sports",
    "The cricket team won by six wickets in the final over",
    "The referee made a controversial decision during the match",
    "The soccer player transferred to a new club for record fee",
    "The sports stadium will host the championship game next week",
    "The athlete won three medals at the national competition",
    "The team captain led the squad to an impressive victory",
    "The marathon runner completed the race in record time",
    "The boxing match was stopped in the eighth round",
    "The hockey team defeated their rivals in overtime",
    "The baseball player hit a home run in the ninth inning",
    "The swimmer qualified for the olympics with a strong performance",
    "The rugby match was played in difficult weather conditions",
    "The tournament bracket was announced for the competition",
    "The player injured during practice will miss the next game",
    "The sports league announced new rules for the season",
    "The team's winning streak reached ten consecutive games",
    "The golf tournament concluded with a thrilling final round",
]

# politics data
POLITICS_DATA = [
    "The prime minister announced new economic policies in parliament today",
    "The election results showed a clear majority for the ruling party",
    "The government passed legislation on healthcare reform",
    "The senator criticized the administration's foreign policy decisions",
    "The presidential candidate outlined their campaign platform",
    "The parliament voted on the new budget proposal",
    "The political party held a rally in the capital city",
    "The diplomat negotiated a peace treaty between the nations",
    "The minister resigned amid corruption allegations",
    "The supreme court ruled on the constitutional amendment",
    "The opposition leader called for transparency in governance",
    "The legislative session addressed climate change policies",
    "The mayor implemented new urban development plans",
    "The governor signed the bill into law yesterday",
    "The political debate focused on immigration reform",
    "The congress approved funding for infrastructure projects",
    "The ambassador represented the country at the summit",
    "The referendum results reflected public opinion on the issue",
    "The cabinet meeting discussed national security concerns",
    "The bilateral talks aimed to strengthen diplomatic relations",
    "The parliament member proposed amendments to the legislation",
    "The election campaign focused on economic issues",
    "The government spokesperson addressed media questions",
    "The coalition government faced challenges in policy implementation",
    "The political analyst predicted changes in voter preferences",
]

class FeatureExtractor:
    """
    this class is used to extract features from the text
    """
    
    def __init__(self):
        # all vocabulary
        self.vocabulary = set()
        
        # idf scores
        self.idf_scores = {}
        
    def tokenize(self, text):
        """
        this function tokenizes the text
        """
        text = text.lower()
        # \\b matches word boundaries, \\w+ matches one or more word characters
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def get_ngrams(self, tokens, n):
        """
        this function creates ngrams from the tokens
        """
        ngrams = []
        # We stop at len(tokens) - n + 1 to avoid going past the end
        for i in range(len(tokens) - n + 1):
            # Take n consecutive tokens starting at position i
            # Join them with spaces to form the n-gram
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def bag_of_words(self, text):
        """
        this function converts text to a simple word frequency dictionary
        """
        tokens = self.tokenize(text)
        # Counter automatically counts how many times each token appears
        bow = Counter(tokens)
        # Convert Counter to regular dictionary
        return dict(bow)
    
    def compute_idf(self, documents):
        """
        this function computes idf scores for all words

        Formula: IDF(word) = log(total documents / (1 + documents containing word))
        The +1 prevents division by zero.
        """
        # Count how many documents each word appears in
        doc_freq = defaultdict(int)
        total_docs = len(documents)
        
        # Go through each document
        for doc in documents:
            # get unique words in this document
            tokens = set(self.tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1
                self.vocabulary.add(token)
        
        # compute idf scores
        for term in doc_freq:
            self.idf_scores[term] = math.log(total_docs / (1 + doc_freq[term]))
    
    def tfidf(self, text):
        """
        this function computes tfidf scores for all words
        """
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        
        # normalize by document length
        doc_length = len(tokens)
        tfidf_vec = {}
        
        for term, count in tf.items():
            # calculate term frequency as proportion of document
            term_freq = count / doc_length
            
            # get the idf score for this term (0 if we haven't seen it before)
            idf = self.idf_scores.get(term, 0)
            
            # multiply tf by idf to get the final score
            tfidf_vec[term] = term_freq * idf
        
        return tfidf_vec


class NaiveBayesClassifier:
    """
    Naive Bayes classifier - a probabilistic approach to classification.
    
    Uses Bayes' theorem:
    P(class|words) = P(class) × P(words|class) / P(words)
    """
    
    def __init__(self):
        # dictionary to count how many times each word appears in each class
        self.class_word_counts = {}
        
        # total number of words in each class
        self.class_totals = {}
        
        # prior probability of each class (how common is each class?)
        self.class_priors = {}
        
        # all unique words we've seen
        self.vocabulary = set()
    
    def train(self, X_train, y_train):
        """
        this function learns word patterns from training data
        """
        # initialize counters for each class
        for label in set(y_train):
            # defaultdict gives us 0 for words we haven't seen instead of an error
            self.class_word_counts[label] = defaultdict(int)
            self.class_totals[label] = 0
        
        # count how many times each word appears in each class
        for features, label in zip(X_train, y_train):
            # features is a dictionary like {"team": 2, "won": 1}
            for word, count in features.items():
                # add this word's count to the class total
                self.class_word_counts[label][word] += count
                self.class_totals[label] += count
                # remember this word for our vocabulary
                self.vocabulary.add(word)
        
        # calculate prior probabilities
        # P(sports) = number of sports docs / total docs
        total_docs = len(y_train)
        class_counts = Counter(y_train)
        for label in class_counts:
            self.class_priors[label] = class_counts[label] / total_docs
    
    def predict(self, features):
        """
        this function predicts which class a document belongs to based on its features
        """
        scores = {}
        vocab_size = len(self.vocabulary)
        
        # calculate score for each possible class
        for label in self.class_priors:
            # start with the prior probability (baseline likelihood of this class)
            # we use log to avoid very small numbers that computers can't handle
            log_prob = math.log(self.class_priors[label])
            
            # add likelihood for each word with laplace smoothing
            # laplace smoothing: add 1 to numerator and vocab_size to denominator
            # this prevents zero probabilities for words we haven't seen
            for word, count in features.items():
                # how many times did we see this word in this class?
                word_count = self.class_word_counts[label].get(word, 0)
                
                # calculate probability with smoothing
                # P(word|class) = (word count in class + 1) / (total words in class + vocab size)
                # the +1 and +vocab_size ensure we never get zero probability
                word_prob = (word_count + 1) / (self.class_totals[label] + vocab_size)
                
                # multiply this word's probability (using log: multiply becomes add)
                # we multiply by count because if word appears twice, it should count twice
                log_prob += count * math.log(word_prob)
            
            scores[label] = log_prob
        
        # return the class with the highest score
        return max(scores, key=scores.get)

class KNNClassifier:
    """
    this class implements the k-nearest neighbors algorithm
    """
    
    def __init__(self, k=3):
        # how many neighbors to consider
        # we usually use an odd number to avoid ties
        self.k = k
        
        # Store all training data (KNN doesn't really "train", it just remembers everything)
        self.X_train = []
        self.y_train = []
    
    def train(self, X_train, y_train):
        """
        For KNN, training just means storing the data.
        We don't build a model - we use the raw training data directly during prediction.
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def cosine_similarity(self, vec1, vec2):
        """
        Formula: dot_product(A, B) / (magnitude(A) × magnitude(B))
        """
        # find words that appear in both documents
        common = set(vec1.keys()) & set(vec2.keys())
        
        # if no words in common, they're completely dissimilar
        if not common:
            return 0.0
        
        # calculate dot product: sum of (value in vec1) × (value in vec2) for common words
        # for example, if "team" has value 0.5 in both, it contributes 0.5 × 0.5 = 0.25
        dot_product = sum(vec1[k] * vec2[k] for k in common)
        
        # calculate magnitude of each vector
        # magnitude is the square root of the sum of squared values
        # this is like the length of the vector in multidimensional space
        mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        # avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # final similarity score
        return dot_product / (mag1 * mag2)
    
    def predict(self, features):
        """
        Classify a document by finding its k nearest neighbors.
        """
        # calculate similarity to every training document
        distances = []
        for train_features, train_label in zip(self.X_train, self.y_train):
            # higher similarity = closer neighbor
            sim = self.cosine_similarity(features, train_features)
            distances.append((sim, train_label))
        
        # sort by similarity (highest first)
        # reverse=True means we want the most similar documents first
        distances.sort(reverse=True, key=lambda x: x[0])
        
        # take the k closest neighbors
        k_nearest = distances[:self.k]
        
        # do a majority vote: which label appears most often?
        # for example, if k=5 and we have ['sports', 'sports', 'politics', 'sports', 'sports'],
        # 'sports' appears 4 times and wins
        votes = Counter([label for _, label in k_nearest])
        return votes.most_common(1)[0][0]

class LogisticRegressionClassifier:
    """
    Logistic Regression - a linear classifier with a sigmoid activation function.
    """
    
    def __init__(self, learning_rate=0.01, epochs=100):
        # how fast we adjust weights during training
        # too high = overshoots the optimal solution
        # Too low = takes forever to converge
        self.learning_rate = learning_rate
        
        # How many times to go through the training data
        self.epochs = epochs
        
        # Dictionary storing the weight for each word/feature
        self.weights = {}
        
        # Bias term - a baseline prediction before considering any words
        self.bias = 0
        
        # The two classes we're trying to distinguish
        self.classes = []
    
    def sigmoid(self, z):
        """
        Formula: 1 / (1 + e^(-z))
        """
        # clip to range [-500, 500] to prevent overflow
        return 1 / (1 + math.exp(-max(min(z, 500), -500)))
    
    def train(self, X_train, y_train):
        """
        Train using gradient descent
        """
        # Figure out what our two classes are
        self.classes = sorted(set(y_train))
        
        # convert labels to binary (0 or 1)
        # we arbitrarily pick the second class as 1 and the first as 0
        y_binary = [1 if y == self.classes[1] else 0 for y in y_train]
        
        # collect all features we've seen across all documents
        all_features = set()
        for features in X_train:
            all_features.update(features.keys())
        
        # initialize all weights to 0 (neutral starting point)
        for feature in all_features:
            self.weights[feature] = 0.0
        
        # gradient descent: repeatedly adjust weights
        for epoch in range(self.epochs):
            # go through each training example
            for features, y_true in zip(X_train, y_binary):
                # calculate our prediction
                # z = bias + sum of (weight × feature_value) for all features
                z = self.bias
                for feature, value in features.items():
                    z += self.weights.get(feature, 0) * value
                
                # apply sigmoid to get probability
                y_pred = self.sigmoid(z)
                
                # how wrong were we?
                error = y_true - y_pred
                
                # update bias to reduce error
                self.bias += self.learning_rate * error
                
                # Update each feature weight
                # Features that appear more strongly should have bigger updates
                for feature, value in features.items():
                    self.weights[feature] += self.learning_rate * error * value
    
    def predict(self, features):
        """
        Make a prediction by calculating the weighted sum and applying sigmoid.
        """
        # calculate z = bias + weighted sum of features
        z = self.bias
        for feature, value in features.items():
            z += self.weights.get(feature, 0) * value
        
        # apply sigmoid to get probability
        prob = self.sigmoid(z)
        
        # if probability >= 0.5, predict class 1, otherwise class 0
        return self.classes[1] if prob >= 0.5 else self.classes[0]


def split_data(X, y, train_ratio=0.8):
    """
    split our data into training and testing sets
    """
    n = len(X)
    # calculate where to make the split
    train_size = int(n * train_ratio)
    
    # Return: X_train, X_test, y_train, y_test
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

def evaluate_classifier(classifier, X_test, y_test):
    """
    metrics for our classifier - accuracy, precision, recall, f1
    """
    # get predictions for all test documents
    predictions = [classifier.predict(x) for x in X_test]
    
    # calculate overall accuracy
    correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
    accuracy = correct / len(y_test)
    
    # calculate per-class metrics
    classes = set(y_test)
    metrics = {}
    
    for cls in classes:
        # True Positives: we predicted this class AND it was actually this class
        tp = sum(1 for pred, true in zip(predictions, y_test) if pred == cls and true == cls)
        
        # False Positives: we predicted this class BUT it was actually another class
        fp = sum(1 for pred, true in zip(predictions, y_test) if pred == cls and true != cls)
        
        # False Negatives: we predicted another class BUT it was actually this class
        fn = sum(1 for pred, true in zip(predictions, y_test) if pred != cls and true == cls)
        
        # Precision: of all our predictions for this class, what fraction were right?
        # High precision = we're careful, we don't label things as this class unless we're sure
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: of all actual examples of this class, what fraction did we find?
        # High recall = we're thorough, we find most examples of this class
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 score: harmonic mean of precision and recall
        # High F1 = good balance between precision and recall
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return accuracy, metrics


if __name__ == "__main__":    
    # combine all our data
    all_texts = SPORTS_DATA + POLITICS_DATA
    # create labels: 'sports' for first 25, 'politics' for last 25
    all_labels = ['sports'] * len(SPORTS_DATA) + ['politics'] * len(POLITICS_DATA)
    
    print("EXPERIMENT 1: Naive Bayes with Bag of Words")
    
    # extract Bag of Words features from all documents
    extractor = FeatureExtractor()
    X_bow = [extractor.bag_of_words(text) for text in all_texts]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = split_data(X_bow, all_labels)
    
    # Train the Naive Bayes classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(X_train, y_train)
    
    # Evaluate on test set
    accuracy, metrics = evaluate_classifier(nb_classifier, X_test, y_test)
    
    # Display results
    print(f"Accuracy: {accuracy:.2%}")
    for cls, scores in metrics.items():
        print(f"{cls.capitalize()}:")
        print(f"Precision: {scores['precision']:.2%}")
        print(f"Recall: {scores['recall']:.2%}")
        print(f"F1-Score: {scores['f1']:.2%}")
    

    print("EXPERIMENT 2: K-Nearest Neighbors with TF-IDF")
    
    # first compute IDF scores across all documents
    extractor_tfidf = FeatureExtractor()
    extractor_tfidf.compute_idf(all_texts)
    
    # extract TF-IDF features
    X_tfidf = [extractor_tfidf.tfidf(text) for text in all_texts]
    X_train, X_test, y_train, y_test = split_data(X_tfidf, all_labels)
    
    # Train KNN classifier with k=5
    knn_classifier = KNNClassifier(k=5)
    knn_classifier.train(X_train, y_train)
    
    # Evaluate
    accuracy, metrics = evaluate_classifier(knn_classifier, X_test, y_test)
    
    print(f"Accuracy: {accuracy:.2%}")
    for cls, scores in metrics.items():
        print(f"{cls.capitalize()}:")
        print(f"Precision: {scores['precision']:.2%}")
        print(f"Recall: {scores['recall']:.2%}")
        print(f"F1-Score: {scores['f1']:.2%}")
    
    print("EXPERIMENT 3: Logistic Regression with TF-IDF")
    
    # use same TF-IDF features from experiment 2
    # train logistic regression with learning rate 0.1 for 50 epochs
    lr_classifier = LogisticRegressionClassifier(learning_rate=0.1, epochs=50)
    lr_classifier.train(X_train, y_train)
    
    # evaluate
    accuracy, metrics = evaluate_classifier(lr_classifier, X_test, y_test)
    
    print(f"Accuracy: {accuracy:.2%}")
    for cls, scores in metrics.items():
        print(f"{cls.capitalize()}:")
        print(f"Precision: {scores['precision']:.2%}")
        print(f"Recall: {scores['recall']:.2%}")
        print(f"F1-Score: {scores['f1']:.2%}")
    
    # let the user test the classifier with their own text
    while True:
        text = input("Your text: ").strip()
        
        # check if user wants to quit
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nThank you!")
            break
        
        # skip empty inputs
        if not text:
            continue
        
        features = extractor.bag_of_words(text)
        prediction = nb_classifier.predict(features)
        print(f"Predicted category: {prediction.upper()}\n")
