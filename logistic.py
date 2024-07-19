import nltk
from nltk import word_tokenize, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
from collections import Counter
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')

df = load_dataset("rohitsaxena/MENSA")
dataset = df["test"]

def extract_features(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    
    # Lexical diversity
    lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0
    
    # Named entities (simple approximation)
    named_entities = sum(1 for word, tag in pos_tags if tag in ['NNP', 'NNPS'])
    
    # Sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(sentence)['compound']
    
    # Readability
    readability = flesch_reading_ease(sentence)
    
    # POS tag distribution
    pos_dist = Counter(tag for word, tag in pos_tags)
    
    # Sentence length
    sentence_length = len(tokens)
    
    # Verb tense
    verb_tenses = [tag for word, tag in pos_tags if tag.startswith('VB')]
    past_tense = sum(1 for tag in verb_tenses if tag in ['VBD', 'VBN'])
    present_tense = sum(1 for tag in verb_tenses if tag in ['VB', 'VBP', 'VBZ'])
    
    features = {
        'lexical_diversity': lexical_diversity,
        'named_entities': named_entities,
        'sentiment': sentiment,
        'readability': readability,
        'sentence_length': sentence_length,
        'past_tense_ratio': past_tense / len(verb_tenses) if verb_tenses else 0,
        'present_tense_ratio': present_tense / len(verb_tenses) if verb_tenses else 0,
    }
    
    # Add POS tag distribution features
    for tag in ['NN', 'NNP', 'VB', 'JJ', 'RB', 'IN', 'DT', 'PRP']:
        features[f'{tag}_ratio'] = pos_dist[tag] / len(pos_tags) if pos_tags else 0
    
    return features


sentences = []
labels = []

for scene_sentences, scene_labels in zip(dataset['scenes'], dataset['labels']):
    sentences.extend(scene_sentences)
    labels.extend(scene_labels)

# Extract features
features_list = [extract_features(sentence) for sentence in sentences]
features_df = pd.DataFrame(features_list)




X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42,  max_iter=9000, solver='liblinear', C=0.1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


feature_importance = pd.DataFrame({'feature': features_df.columns, 'importance': model.coef_[0]})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))


y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.4f}")


def classify_sentence(sentence, threshold=optimal_threshold):
    features = extract_features(sentence)
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    probability = model.predict_proba(features_scaled)[0]
    return "Salient" if probability[1] >= threshold else "Non-salient", probability


example_sentence = "This is an example sentence to classify."
classification, probability = classify_sentence(example_sentence)
print(f"\nExample sentence: '{example_sentence}'")
print(f"Classification: {classification}")
print(f"Probability: Salient {probability[1]:.2f}, Non-salient {probability[0]:.2f}")

