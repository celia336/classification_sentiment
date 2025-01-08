# src/data_preprocessing.py

import os
import pandas as pd
from utils import clean_text

def preprocess_data(input_file, output_csv):
    """
    Cette fonction effectue le prétraitement des tweets et les exporte dans un fichier CSV.
    Elle nettoie le texte des tweets et les enregistre dans un fichier CSV, tout en conservant les labels existants.
    
    :param input_file: Le chemin vers le fichier d'entrée contenant les tweets et leurs labels.
    :param output_csv: Le chemin où le fichier CSV nettoyé sera sauvegardé.
    """
    # Vérifier si le fichier d'entrée existe
    if not os.path.exists(input_file):
        print(f"Erreur : Le fichier {input_file} est introuvable.")
        return
    
    # Charger les données depuis le fichier texte (tweets.txt)
    print(f"Chargement des tweets depuis {input_file}...")
    try:
        # On suppose que les tweets sont sous une forme de texte brut avec le label à la fin
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        if len(lines) == 0:
            print(f"Erreur : Le fichier {input_file} est vide.")
            return
        
        # Préparer les listes pour les textes et les labels
        texts = []
        labels = []
        
        for line in lines:
            # Séparer le texte et le label à partir du dernier espace
            parts = line.rsplit(' ', 1)  # Sépare à partir du dernier espace
            if len(parts) == 2:
                text = parts[0].strip()  # Texte sans espaces autour
                label = parts[1].strip()  # Label sans espaces autour
                texts.append(text)
                labels.append(label)
        
        # Créer un DataFrame avec les colonnes 'text' et 'label'
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Vérification du contenu du DataFrame
        print(f"Quelques lignes chargées depuis le fichier texte :\n{df.head()}")
        
        # Appliquer le nettoyage sur le texte des tweets
        print("Nettoyage du texte des tweets...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Vérifier quelques lignes après nettoyage
        print(f"Quelques tweets nettoyés :\n{df[['text', 'cleaned_text']].head()}")

        # Sauvegarder le DataFrame dans un fichier CSV
        print(f"Exportation des données vers {output_csv}...")
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Le fichier a été sauvegardé sous {output_csv}.")
        
    except Exception as e:
        print(f"Erreur lors de la lecture ou de l'écriture du fichier : {e}")

# Exemple d'appel de la fonction
input_file = 'data/raw/tweets.txt'  # Chemin vers votre fichier d'entrée contenant les tweets et labels
output_csv = 'data/processed/cleaned_tweets.csv'  # Chemin où vous voulez enregistrer le fichier CSV
preprocess_data(input_file, output_csv)
