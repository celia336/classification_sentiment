# src/data_preprocessing.py

import os
import pandas as pd
from utils import clean_text

def preprocess_data(input_file, output_csv):
    """
    Cette fonction effectue le prétraitement des tweets et les exporte dans un fichier CSV.
    Elle nettoie le texte des tweets et les enregistre dans un fichier CSV.
    
    :param input_file: Le chemin vers le fichier d'entrée contenant les tweets bruts.
    :param output_csv: Le chemin où le fichier CSV nettoyé sera sauvegardé.
    """
    # Vérifier si le fichier d'entrée existe
    if not os.path.exists(input_file):
        print(f"Erreur : Le fichier {input_file} est introuvable.")
        return
    
    # Charger les données depuis le fichier texte (tweets.txt)
    print(f"Chargement des tweets depuis {input_file}...")
    try:
        # On suppose que les tweets sont sous une forme de texte brut
        with open(input_file, 'r', encoding='utf-8') as file:
            tweets = file.readlines()
        
        if len(tweets) == 0:
            print(f"Erreur : Le fichier {input_file} est vide.")
            return
        
        # Créer un DataFrame avec une colonne 'text' pour les tweets
        df = pd.DataFrame(tweets, columns=['text'])
        
        # Vérifier les 5 premières lignes pour s'assurer du format des données
        print(f"Quelques tweets chargés depuis {input_file} :\n{df.head()}")

        # Ajouter une colonne 'label' (ici, vous pouvez choisir un label par défaut ou le laisser vide)
        df['label'] = 'neutral'  # Remplacez par des labels réels si disponible
        print(f"Structure du DataFrame après ajout de la colonne 'label' :\n{df.head()}")

        # Appliquer le nettoyage sur le texte des tweets
        print("Nettoyage du texte des tweets...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Vérifier quelques lignes après nettoyage
        print(f"Quelques tweets nettoyés :\n{df[['text', 'cleaned_text']].head()}")

        # Sauvegarder le DataFrame dans un fichier CSV
        print(f"Exportation des données vers {output_csv}...")
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Le fichier cleaned_tweets.csv a été sauvegardé sous {output_csv}.")
        
    except Exception as e:
        print(f"Erreur lors de la lecture ou de l'écriture du fichier : {e}")

# Exemple d'appel de la fonction
input_file = 'data/raw/tweets.txt'  # Vérifiez que ce chemin est correct
output_csv = 'data/processed/cleaned_tweets.csv'
preprocess_data(input_file, output_csv)
