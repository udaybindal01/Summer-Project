import nltk
from nltk import word_tokenize, pos_tag
from collections import Counter
from datasets import load_dataset
import numpy as np


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


df = load_dataset("rohitsaxena/MENSA")
dataset = df["test"]

def get_pos_ngrams(sentences, n=3):
 
    ngrams = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        pos_tags = [tag for word, tag in pos_tag(tokens)]
        ngrams.extend(list(nltk.ngrams(pos_tags, n)))
    return ngrams

def analyze_pos_patterns(dataset, n=3):
    salient_scenes = []
    non_salient_scenes = []
    
    for scene_sentences, scene_labels in zip(dataset['scenes'], dataset['labels']):
        salient_sentences = [s for s, l in zip(scene_sentences, scene_labels) if l == 1]
        non_salient_sentences = [s for s, l in zip(scene_sentences, scene_labels) if l == 0]
        
        salient_scenes.extend(salient_sentences)
        non_salient_scenes.extend(non_salient_sentences)
    
    salient_ngrams = get_pos_ngrams(salient_scenes, n)
    non_salient_ngrams = get_pos_ngrams(non_salient_scenes, n)
    
    salient_counts = Counter(salient_ngrams)
    non_salient_counts = Counter(non_salient_ngrams)
    
    total_salient = sum(salient_counts.values())
    total_non_salient = sum(non_salient_counts.values())
    
    salient_freq = {k: v/total_salient for k, v in salient_counts.items()}
    non_salient_freq = {k: v/total_non_salient for k, v in non_salient_counts.items()}
    
    all_ngrams = set(salient_freq.keys()) | set(non_salient_freq.keys())
    
    differences = {}
    for ngram in all_ngrams:
        salient_freq_value = salient_freq.get(ngram, 0)
        non_salient_freq_value = non_salient_freq.get(ngram, 0)
        diff = salient_freq_value - non_salient_freq_value
        differences[ngram] = diff
    
    return differences

# Analyze POS patterns
pos_differences = analyze_pos_patterns(dataset)


sorted_patterns = sorted(pos_differences.items(), key=lambda x: abs(x[1]), reverse=True)

# Print top 20 most different patterns
print("Top 20 POS patterns with largest difference (Salient - Non-Salient):")
for pattern, diff in sorted_patterns[:20]:
    print(f"{' '.join(pattern)}: {diff:.4f}")

# Analyze verb tense distribution
def get_tense(pos_tag):
    if pos_tag.startswith('VB'):
        if pos_tag == 'VBD' or pos_tag == 'VBN':
            return 'Past'
        elif pos_tag == 'VBG':
            return 'Present Participle'
        elif pos_tag in ['VB', 'VBP', 'VBZ']:
            return 'Present'
    return 'Other'

salient_tenses = []
non_salient_tenses = []

for scene_sentences, scene_labels in zip(dataset['scenes'], dataset['labels']):
    for sentence, label in zip(scene_sentences, scene_labels):
        tokens = word_tokenize(sentence)
        pos_tags = [tag for word, tag in pos_tag(tokens)]
        tenses = [get_tense(tag) for tag in pos_tags]
        if label == 1:
            salient_tenses.extend(tenses)
        else:
            non_salient_tenses.extend(tenses)

salient_tense_dist = Counter(salient_tenses)
non_salient_tense_dist = Counter(non_salient_tenses)

print("\nTense Distribution:")
print("Tense\t\tSalient\t\tNon-Salient")
for tense in set(salient_tense_dist.keys()) | set(non_salient_tense_dist.keys()):
    salient_percent = salient_tense_dist[tense] / sum(salient_tense_dist.values()) * 100
    non_salient_percent = non_salient_tense_dist[tense] / sum(non_salient_tense_dist.values()) * 100
    print(f"{tense}\t\t{salient_percent:.2f}%\t\t{non_salient_percent:.2f}%")