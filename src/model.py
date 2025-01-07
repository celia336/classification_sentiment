#src/model.py

import torch
import torch.nn as nn
import torch.optim as optim

# Définition d'un modèle de réseau de neurones pour la classification de sentiment
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, dropout_prob=0.3):
        super(SentimentModel, self).__init__()

        # Définir l'Embedding avec vocab_size et embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Première couche linéaire
        self.fc1 = nn.Linear(embedding_dim, hidden_size)  # Assurez-vous que l'input est de taille embedding_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)  # Dropout pour éviter le surapprentissage

        # Deuxième couche linéaire
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Couche de sortie
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Passer par l'Embedding
        x = self.embedding(x)

        # Moyenne des embeddings pour chaque séquence de mots (pooling)
        x = torch.mean(x, dim=1)  # Moyenne des embeddings sur la dimension des mots

        # Passer par la première couche
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Passer par la deuxième couche
        x = self.fc2(x)
        x = self.relu(x)

        # Passer par la couche de sortie
        x = self.fc3(x)

        return x
