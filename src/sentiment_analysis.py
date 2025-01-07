#src/sentiment_analysis.py

import pandas as pd
import torch
from model import SentimentModel  # Utilisez SentimentModel
import pickle
from utils import clean_text  # Fonction de nettoyage de texte

# Fonction pour prédire le sentiment d'un tweet
def predict_sentiment(tweet, model, word_to_idx, input_size=50):
    # Nettoyage du texte
    cleaned_tweet = clean_text(tweet)
    # Créer un vecteur d'entrée pour le modèle
    tokenized_tweet = [word_to_idx.get(word, 0) for word in cleaned_tweet.split()]
    tokenized_tweet = tokenized_tweet[:input_size] + [0] * (input_size - len(tokenized_tweet))  # Compléter le vecteur
    tweet_tensor = torch.tensor([tokenized_tweet], dtype=torch.long)
    
    with torch.no_grad():
        output = model(tweet_tensor)  # Passer le tweet à travers le modèle
        _, predicted = torch.max(output, 1)  # Prendre la classe avec la plus haute probabilité
    
    # Retourner le sentiment prédit
    sentiment = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment.get(predicted.item(), 'Unknown')

# Fonction principale pour analyser les sentiments des tweets
def analyze_sentiments(input_file='data/processed/cleaned_tweets.csv'):
    try:
        # Lire le fichier CSV contenant les tweets
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Le fichier {input_file} n'est pas trouvé. Vérifiez son emplacement.")
        return

    # Charger le vocabulaire (word_to_idx) et label_to_idx depuis l'entraînement
    with open('data/external/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    
    with open('data/external/label_to_idx.pkl', 'rb') as f:
        label_to_idx = pickle.load(f)

    # Calculer la taille du vocabulaire et le nombre de classes de manière dynamique
    vocab_size = len(word_to_idx) + 1  # Taille du vocabulaire (ajoutez 1 pour le token 'PAD')
    output_size = len(label_to_idx)  # Nombre de classes

    # Définir les autres dimensions du modèle
    embedding_dim = 128  # Dimension des embeddings
    hidden_size = 128  # Taille de la couche cachée

    # Initialisation du modèle
    model = SentimentModel(vocab_size=vocab_size, embedding_dim=embedding_dim,
                           hidden_size=hidden_size, output_size=output_size)  # Ajustez les dimensions

    try:
        model.load_state_dict(torch.load('data/external/sentiment_model.pth'))
    except FileNotFoundError:
        print("Le modèle sentiment_model.pth est introuvable. Vérifiez son emplacement.")
        return
    except RuntimeError as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    model.eval()  # Passer le modèle en mode évaluation

    # Analyser les sentiments des tweets
    if 'text' not in df.columns:
        print("Le fichier CSV doit contenir une colonne 'text'.")
        return

    # Appliquer la prédiction sur chaque tweet
    df['sentiment'] = df['text'].apply(lambda tweet: predict_sentiment(tweet, model, word_to_idx))

    # Sauvegarder les résultats
    output_csv = 'data/processed/tweets_sentiment.csv'
    df[['text', 'sentiment']].to_csv(output_csv, index=False)
    print(f"Les résultats de l'analyse des sentiments ont été exportés vers {output_csv}")

# Appel de la fonction
if __name__ == "__main__":
    analyze_sentiments(input_file='data/processed/cleaned_tweets.csv')
