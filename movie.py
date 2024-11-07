
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from nltk.corpus import movie_reviews
import random

# Téléchargement des données NLTK (pour la première utilisation)
nltk.download('movie_reviews')
nltk.download('punkt')

# Chargement des données de critiques de films
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)  # Mélange des données

# Préparation des données
texts = [" ".join(words) for words, label in documents]
labels = [label for words, label in documents]

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Création du pipeline : vectorisation TF-IDF + régression logistique
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Entraînement du modèle
pipeline.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Évaluation du modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
