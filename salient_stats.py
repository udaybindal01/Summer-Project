import pandas as pd
import numpy as np
from collections import Counter
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
import spacy
from datasets import load_dataset
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('vader_lexicon')


nlp = spacy.load('en_core_web_sm')

df = load_dataset("rohitsaxena/MENSA")
df = df["test"]

def analyze_sentences(sentences):
    analysis = {
        'avg_length': [],
        'lexical_diversity': [],
        'pos_distribution': [],
        'named_entities': [],
        'sentiment': [],
        'readability': []
    }
    
    for sentence in sentences:
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        
        # Sentence length
        analysis['avg_length'].append(len(tokens))
        
        # Lexical diversity (Type-Token Ratio)
        analysis['lexical_diversity'].append(len(set(tokens)) / len(tokens))
        
        # POS distribution
        pos_dist = Counter(tag for word, tag in pos_tag(tokens))
        analysis['pos_distribution'].append(pos_dist)
        
        # Named Entities
        named_entities = ne_chunk(pos_tag(tokens))
        ne_count = sum(1 for chunk in named_entities if hasattr(chunk, 'label'))
        analysis['named_entities'].append(ne_count)
        
        # Sentiment
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(sentence)
        analysis['sentiment'].append(sentiment['compound'])
        
        # Readability
        analysis['readability'].append(flesch_reading_ease(sentence))
    
    return analysis

def analyze_scene(scene, scene_labels):
        
    salient_sentences = []
    non_salient_sentences = []
    
    for sentence, label in zip(scene, scene_labels):
        if label == 1:
            salient_sentences.append(sentence)
        else:
            non_salient_sentences.append(sentence)
    
    return salient_sentences, non_salient_sentences


all_salient_sentences = []
all_non_salient_sentences = []


for scene_summary, scene_labels in zip(df['scenes'], df['labels']):
    # print(scene_summary)
    salient, non_salient = analyze_scene(scene_summary, scene_labels)
    all_salient_sentences.extend(salient)
    all_non_salient_sentences.extend(non_salient)

# Analyze both groups
salient_analysis = analyze_sentences(all_salient_sentences)
non_salient_analysis = analyze_sentences(all_non_salient_sentences)


print("Salient vs Non-Salient Comparison:")
print(f"Average Length: {np.mean(salient_analysis['avg_length']):.2f} vs {np.mean(non_salient_analysis['avg_length']):.2f}")
print(f"Average Lexical Diversity: {np.mean(salient_analysis['lexical_diversity']):.2f} vs {np.mean(non_salient_analysis['lexical_diversity']):.2f}")
print(f"Average Named Entities: {np.mean(salient_analysis['named_entities']):.2f} vs {np.mean(non_salient_analysis['named_entities']):.2f}")
print(f"Average Sentiment: {np.mean(salient_analysis['sentiment']):.2f} vs {np.mean(non_salient_analysis['sentiment']):.2f}")
print(f"Average Readability: {np.mean(salient_analysis['readability']):.2f} vs {np.mean(non_salient_analysis['readability']):.2f}")

# POS Distribution
salient_pos = Counter()
for pos in salient_analysis['pos_distribution']:
    salient_pos.update(pos)
non_salient_pos = Counter()
for pos in non_salient_analysis['pos_distribution']:
    non_salient_pos.update(pos)

print("\nPOS Distribution (Salient vs Non-Salient):")
for pos in set(salient_pos.keys()) | set(non_salient_pos.keys()):
    print(f"{pos}: {salient_pos[pos] / sum(salient_pos.values()):.2f} vs {non_salient_pos[pos] / sum(non_salient_pos.values()):.2f}")

# Tense analysis using spaCy
def get_tense(sentence):
    doc = nlp(sentence)
    tenses = [token.morph.get('Tense') for token in doc if token.pos_ == 'VERB']
    return Counter(tense[0] for tense in tenses if tense)

salient_tenses = Counter()
for scene in all_salient_sentences:
    salient_tenses.update(get_tense(scene))

non_salient_tenses = Counter()
for scene in all_non_salient_sentences:
    non_salient_tenses.update(get_tense(scene))

print("\nTense Distribution (Salient vs Non-Salient):")
for tense in set(salient_tenses.keys()) | set(non_salient_tenses.keys()):
    print(f"{tense}: {salient_tenses[tense] / sum(salient_tenses.values()):.2f} vs {non_salient_tenses[tense] / sum(non_salient_tenses.values()):.2f}")


