#src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
from model import SentimentModel  # Importer le modèle que nous avons défini dans model.py
import pickle  # Importation du module pickle

# Téléchargement des ressources NLTK si nécessaire
nltk.download('stopwords')
nltk.download('punkt')

# Configuration de l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len=50):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Conversion du texte en indices numériques
        encoded_text = [self.word_to_idx.get(word, 0) for word in text.split()]
        padded_text = encoded_text[:self.max_len] + [0] * (self.max_len - len(encoded_text))

        return torch.tensor(padded_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# Fonction de prétraitement du texte
def preprocess_text(text):
    stop_words = set(stopwords.words('french'))
    words = word_tokenize(text.lower())
    return " ".join(word for word in words if word.isalnum() and word not in stop_words)


# Entraînement du modèle
def train_model(data_path='data/processed/cleaned_tweets.csv', model_path='data/external/sentiment_model.pth',
                batch_size=32, num_epochs=5, learning_rate=0.001):
    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        print(f"Le fichier {data_path} n'existe pas.")
        return

    # Chargement des données
    df = pd.read_csv(data_path)
    # Aucune opération de nettoyage ici, car les données sont déjà nettoyées dans data_preprocessing.py

    # Construction du vocabulaire
    all_words = set(word for text in df['cleaned_text'] for word in text.split())
    word_to_idx = {word: idx + 1 for idx, word in enumerate(all_words)}

    # Sauvegarder le vocabulaire
    os.makedirs('data/external', exist_ok=True)
    with open('data/external/word_to_idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f)
    print("Vocabulaire sauvegardé dans 'data/external/word_to_idx.pkl'.")

    # Encodage des labels
    label_to_idx = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['encoded_label'] = df['label'].map(label_to_idx)

    # Sauvegarder label_to_idx
    with open('data/external/label_to_idx.pkl', 'wb') as f:
        pickle.dump(label_to_idx, f)
    print("Label_to_idx sauvegardé dans 'data/external/label_to_idx.pkl'.")

    # Séparation en ensembles d'entraînement et de test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned_text'], df['encoded_label'], test_size=0.2, random_state=42
    )

    # Création des jeux de données
    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), word_to_idx)
    test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), word_to_idx)

    # Création des DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialisation des paramètres
    vocab_size = len(word_to_idx) + 1  # Taille du vocabulaire
    embedding_dim = 128  # Dimension des embeddings
    hidden_dim = 128  # Taille de la couche cachée
    output_dim = len(label_to_idx)  # Nombre de classes (sentiments)

    # Initialisation du modèle
    model = SentimentModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement du modèle
    for epoch in range(num_epochs):
        model.train()
        for texts_batch, labels_batch in train_loader:
            texts_batch = texts_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} terminée avec perte: {loss.item()}")

    # Évaluation du modèle
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts_batch, labels_batch in test_loader:
            texts_batch = texts_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(texts_batch)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    print(f"Précision sur l'ensemble de test : {100 * correct / total:.2f}%")

    # Sauvegarde du modèle
    model_save_path = os.path.abspath(model_path)
    torch.save(model.state_dict(), model_save_path)
    print(f"Modèle sauvegardé dans {model_save_path}")


if __name__ == "__main__":
    train_model()
