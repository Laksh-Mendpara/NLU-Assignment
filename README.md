# Natural Language Understanding Assignment

**Course:** CSL 7640 - Natural Language Understanding  
**Name:** Laksh Mendpara  
**Roll Number:** B23CS1037

## 📋 Overview

This repository contains the complete implementation of four Natural Language Understanding problems, covering fundamental NLP techniques from regular expressions to machine learning classifiers. All implementations are done from scratch using only Python standard libraries.

## 📂 Repository Structure

```
NLU-Assignment/
├── B23CS1037_prob1.py          # Reggy++ Chatbot
├── B23CS1037_prob1.log         # Chatbot conversation logs
├── B23CS1037_prob1.txt         # Reflection document
├── B23CS1037_prob2.py          # Byte Pair Encoding
├── B23CS1037_prob3.py          # Sentiment Classifier
├── B23CS1037_prob4.py          # Sports vs Politics Classifier
├── report.tex                  # LaTeX report source
├── report.pdf                  # Compiled report
├── data/                       # Training data files
│   ├── pos.txt                 # Positive sentiment data
│   ├── neg.txt                 # Negative sentiment data
│   └── test_corpus.txt         # BPE test corpus
└── docs/                       # GitHub Pages
    ├── index.html              # Main documentation page
    └── report.pdf              # Detailed report
```

## 🎯 Problems Overview

### Problem 1: Reggy++ Chatbot

A conversational chatbot using regular expressions for:

- Multiple date format parsing (mm-dd-yy, dd/mm/yyyy, dd Month yyyy)
- Age calculation from birthdate
- Mood detection with typo tolerance
- Surname extraction

**Run:**

```bash
python B23CS1037_prob1.py
```

**Features:**

- Supports 6+ date formats
- Handles common typos in mood words
- Interactive conversation flow

### Problem 2: Byte Pair Encoding (BPE)

From-scratch implementation of the BPE tokenization algorithm.

**Run:**

```bash
python B23CS1037_prob2.py k corpus.txt
```

**Example:**

```bash
python B23CS1037_prob2.py 10 data/test_corpus.txt
```

**Features:**

- Character-level vocabulary initialization
- Iterative pair merging
- Subword token generation

### Problem 3: Naive Bayes Sentiment Classifier

Probabilistic sentiment analysis classifier with Laplace smoothing.

**Run:**

```bash
python B23CS1037_prob3.py
```

**Features:**

- Bag-of-Words representation
- Laplace smoothing for unseen words
- Interactive sentiment prediction
- Trained on positive/negative sentence pairs

### Problem 4: Sports vs Politics Text Classifier

Comprehensive ML comparison: Naive Bayes, KNN, and Logistic Regression.

**Run:**

```bash
python B23CS1037_prob4.py
```

**Features:**

- 3 ML algorithms comparison
- Multiple feature representations (BoW, TF-IDF)
- Performance metrics (Accuracy, Precision, Recall, F1)
- Interactive classification mode

## 📊 Results Summary

| Problem   | Technique           | Key Achievement                          |
| --------- | ------------------- | ---------------------------------------- |
| Problem 1 | Regular Expressions | Multi-format parsing with typo tolerance |
| Problem 2 | BPE Algorithm       | Efficient subword tokenization           |
| Problem 3 | Naive Bayes         | High-accuracy sentiment classification   |
| Problem 4 | ML Comparison       | **Naive Bayes: 80%**, KNN: 50%, LR: 30%  |

### Problem 4: Detailed Results

| Method              | Features     | Accuracy   | F1-Score   |
| ------------------- | ------------ | ---------- | ---------- |
| **Naive Bayes**     | Bag of Words | **80.00%** | **88.89%** |
| K-Nearest Neighbors | TF-IDF       | 50.00%     | 66.67%     |
| Logistic Regression | TF-IDF       | 30.00%     | 46.15%     |

## 📖 Documentation

- **GitHub Pages:** [View Documentation](https://laksh-mendpara.github.io/NLU-Assignment/)
- **Detailed Report:** [report.pdf](report.pdf) - 9-page comprehensive analysis
- **Reflection (Problem 1):** [B23CS1037_prob1.txt](B23CS1037_prob1.txt)

## 🚀 Quick Start

### Prerequisites

- Python 3.6 or higher
- No external libraries required (uses only Python standard library)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Laksh-Mendpara/NLU-Assignment.git
cd NLU-Assignment
```

2. Run any problem:

```bash
# Problem 1: Chatbot
python B23CS1037_prob1.py

# Problem 2: BPE (10 merges on test corpus)
python B23CS1037_prob2.py 10 data/test_corpus.txt

# Problem 3: Sentiment Classifier
python B23CS1037_prob3.py

# Problem 4: Sports vs Politics
python B23CS1037_prob4.py
```

## 💡 Key Implementation Details

### Problem 1: Regular Expressions

- **Date Parsing:** Handles multiple formats with century inference
- **Mood Detection:** Regex patterns for 6 mood categories with typo variants
- **Surname Extraction:** Simple word-boundary based extraction

### Problem 2: BPE Algorithm

- **Vocabulary Initialization:** Character-level with `</w>` end-of-word marker
- **Merging Strategy:** Frequency-based pair selection
- **Efficiency:** O(k\*n) where k=merges, n=vocabulary size

### Problem 3: Naive Bayes

- **Laplace Smoothing:** Prevents zero probabilities
- **Log Probabilities:** Avoids numerical underflow
- **Formula:** P(class|words) ∝ P(class) × ∏P(word|class)

### Problem 4: ML Comparison

- **Feature Extraction:** BoW and TF-IDF implementations
- **Classifiers:** Naive Bayes, KNN (k=5), Logistic Regression (LR=0.1, 50 epochs)
- **Dataset:** 50 manually created articles (balanced classes)
- **Evaluation:** 80/20 train-test split

## 🔍 Code Quality

- ✅ **No External Libraries:** Pure Python implementations
- ✅ **Detailed Comments:** Human-like explanations throughout
- ✅ **Modular Design:** Clear separation of concerns
- ✅ **Error Handling:** Graceful handling of edge cases
- ✅ **Interactive Modes:** User-friendly interfaces

## 📝 Assignment Requirements

All deliverables completed:

- ✅ **Problem 1:** Python script, log file (10 runs), reflection text
- ✅ **Problem 2:** Python script with BPE implementation
- ✅ **Problem 3:** Python script with Naive Bayes classifier
- ✅ **Problem 4:** Python script, detailed report (9 pages), GitHub pages

## 🎓 Learning Outcomes

This assignment demonstrates understanding of:

1. Regular expressions for pattern matching
2. Tokenization algorithms (BPE)
3. Probabilistic classifiers (Naive Bayes)
4. Feature engineering (BoW, TF-IDF)
5. ML algorithm comparison and evaluation
6. From-scratch implementations without ML libraries

## 📧 Contact

**Laksh Mendpara**  
Roll Number: B23CS1037  
Course: CSL 7640 - Natural Language Understanding

## 📄 License

This project is created for academic purposes as part of the NLU course assignment.

---

**Note:** All code is original and implemented from scratch without using external NLP/ML libraries, adhering to assignment requirements.
