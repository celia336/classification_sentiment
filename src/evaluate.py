import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from src.model import SentimentModel
from src.utils import clean_text
from src.data_preprocessing import TweetDataset  # Assurez-vous que cette classe est bien définie dans data_preprocessing.py

# Chargement du modèle
model = SentimentModel(input_size=100, hidden_size=128, output_size=2)  # Adapter selon le nombre de classes
model.load_state_dict(torch.load('sentiment_model.pth'))
model.eval()  # Passer en mode évaluation

# Chargement des données de test
df = pd.read_csv('data/processed/cleaned_tweets.csv')
df['text'] = df['text'].apply(clean_text)
texts = df['text'].values
labels = df['label'].values

# Tokeniser les textes (Remplacer par une tokenisation plus avancée si nécessaire)
tokenized_texts = [len(text.split()) for text in texts]  # Remplacez par votre propre tokenisation

# Préparer le DataLoader
test_dataset = TweetDataset(tokenized_texts, labels)
test_loader = DataLoader(test_dataset, batch_size=32)

# Prédiction et évaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        # Assurez-vous que les entrées sont au bon format et de type float
        texts = texts.float()
        outputs = model(texts)  # Passer les données dans le modèle
        _, predicted = torch.max(outputs, 1)  # Récupérer la classe avec la probabilité la plus élevée
        all_preds.extend(predicted.cpu().numpy())  # Sauvegarder les prédictions
        all_labels.extend(labels.cpu().numpy())  # Sauvegarder les étiquettes réelles

# Calcul de l'exactitude
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Rapport de classification
print("Classification Report:")
print(classification_report(all_labels, all_preds))

# Optionnel: Sauvegarder les résultats d'évaluation dans un fichier
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(all_labels, all_preds))
