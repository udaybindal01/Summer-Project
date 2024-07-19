import pickle
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from textblob import TextBlob
import spacy
from datasets import load_dataset
import string


nlp = spacy.load("en_core_web_sm")

# Download NLTK stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

# stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    
    doc = nlp(text)
    tokens = word_tokenize(text.lower())
    # tokens = [word for word in tokens if word.isalpha() and word not in stop_words and word not in punctuation]
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

    return ' '.join(tokens)

# Function for lexical analysis
def lexical_analysis(texts):
    token_counter = Counter()
    word_lengths = []
    
    for text in texts:
        tokens = word_tokenize(text)
        token_counter.update(tokens)
        word_lengths.extend([len(token) for token in tokens])
    
    vocab_size = len(token_counter)
    avg_word_length = np.mean(word_lengths)
    
    return token_counter, vocab_size, avg_word_length

# Function for syntactic analysis
def syntactic_analysis(texts):
    pos_counter = Counter()
    sentence_lengths = []
    
    for text in texts:
        doc = nlp(text)
        pos_counter.update([token.pos_ for token in doc])
        sentence_lengths.extend([len(sent) for sent in doc.sents])
    
    avg_sentence_length = np.mean(sentence_lengths)
    return pos_counter, avg_sentence_length

# Function for semantic analysis
def semantic_analysis(texts):
    entity_counter = Counter()
    sentiments = []
    
    for text in texts:
        doc = nlp(text)
        entity_counter.update([ent.label_ for ent in doc.ents])
        sentiment = TextBlob(text).sentiment
        sentiments.append((sentiment.polarity, sentiment.subjectivity))
    
    avg_sentiment = np.mean([s[0] for s in sentiments])
    avg_subjectivity = np.mean([s[1] for s in sentiments])
    return entity_counter, avg_sentiment, avg_subjectivity

# Function for stylistic analysis
# def stylistic_analysis(texts):
#     readability_scores = []
    
#     for text in texts:
#         readability_scores.append(TextBlob(text).readability)
    
#     avg_readability = np.mean(readability_scores)
#     return avg_readability

# Function for bias and fairness analysis
def bias_analysis(texts):
    gender_counter = Counter()
    ethnic_counter = Counter()
    
    gender_words = {
        'male': ['he', 'him', 'his', 'man', 'men'],
        'female': ['she', 'her', 'hers', 'woman', 'women']
    }
    
    ethnic_words = {
        'asian': ['asian', 'chinese', 'japanese', 'korean'],
        'black': ['black', 'african', 'african-american'],
        'white': ['white', 'caucasian', 'european']
    }
    
    for text in texts:
        tokens = word_tokenize(text)
        for token in tokens:
            for gender, words in gender_words.items():
                # print("HERE\n")
                if token.lower() in words:
                    gender_counter[gender] += 1
            # for ethnic, words in ethnic_words.items():
            #     if token.lower() in words:
            #         ethnic_counter[ethnic] += 1
    
    return gender_counter, ethnic_counter

# Load and analyze dataset
def analyze_dataset(dataset):
    scenes = []
    labels = []

    for data in dataset:
        processed_scenes = [preprocess_text(scene) for scene in data["scenes"]]
        scenes.extend(processed_scenes)
        labels.extend(data["labels"])
    print("HERE1\n")
    # Lexical analysis
    token_counter, vocab_size, avg_word_length = lexical_analysis(scenes)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Average word length: {avg_word_length:.2f}")



    # # Syntactic analysis
    pos_counter, avg_sentence_length = syntactic_analysis(scenes)
    print(f"POS distribution: {pos_counter}")
    print(f"Average sentence length: {avg_sentence_length:.2f}")

    # Semantic analysis
    entity_counter, avg_sentiment, avg_subjectivity = semantic_analysis(scenes)
    print(f"Entity distribution: {entity_counter}")
    print(f"Average sentiment polarity: {avg_sentiment:.2f}")
    print(f"Average sentiment subjectivity: {avg_subjectivity:.2f}")

    # # Stylistic analysis
    # avg_readability = stylistic_analysis(scenes)
    # print(f"Average readability score: {avg_readability:.2f}")

    # Bias and fairness analysis
    gender_counter, ethnic_counter = bias_analysis(scenes)
    print(f"Gender word distribution: {gender_counter}")
    print(f"Ethnic word distribution: {ethnic_counter}")

if __name__ == '__main__':
    
    
    dataset = load_dataset("rohitsaxena/MENSA")
    train = dataset["train"]
    # train = train[:100]
    print(len(train))
    validation = dataset["validation"]
    test = dataset["test"]

    # print("Training Dataset Statistics:")
    # analyze_dataset(train)

    print("\nValidation Dataset Statistics:")
    analyze_dataset(validation)

    print("\nTest Dataset Statistics:")
    analyze_dataset(test)
