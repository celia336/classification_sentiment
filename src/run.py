import os
import sys
from data_preprocessing import preprocess_data
from train import train_model
from sentiment_analysis import analyze_sentiments
from dashboard import launch_dashboard

def main(source='api'):
    """
    Point d'entrée principal pour l'application.
    Args:
        source (str): Source des tweets ('api' ou 'file'). Par défaut 'api'.
    """
    # Définir les chemins pour les fichiers
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    raw_tweets_file = os.path.join(raw_data_dir, 'raw_tweets.csv')

    # Vérification ou création des dossiers nécessaires
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    if source == 'file':
        # Si source = file, chercher le fichier 'tweets.txt'
        local_file = os.path.join(raw_data_dir, 'tweets.txt')
        if not os.path.exists(local_file):
            print(f"Erreur : Le fichier {local_file} est introuvable.")
            sys.exit(1)
        print(f"Chargement des tweets depuis {local_file}...")
        preprocess_data(input_file=local_file, output_csv=raw_tweets_file)

    elif source == 'api':
        # Si source = api, exécuter publier_tweets.py
        print("Récupération des tweets via l'API Twitter...")
        os.system(f"python publier_tweets.py")
        if not os.path.exists(raw_tweets_file):
            print(f"Erreur : Le fichier {raw_tweets_file} n'a pas été créé par l'API.")
            sys.exit(1)
    else:
        print("Erreur : La source spécifiée doit être 'api' ou 'file'.")
        sys.exit(1)

    # Étape 2 : Prétraitement des données
    print("Prétraitement des données...")
    preprocess_data(input_file=raw_tweets_file, output_csv=os.path.join(processed_data_dir, 'cleaned_tweets.csv'))

    # Vérifiez si le fichier cleaned_tweets.csv existe avant de passer à l'entraînement
    cleaned_tweets_file = os.path.join(processed_data_dir, 'cleaned_tweets.csv')
    if not os.path.exists(cleaned_tweets_file):
        print(f"Erreur : Le fichier {cleaned_tweets_file} n'a pas été créé lors du prétraitement.")
        sys.exit(1)

    # Étape 3 : Entraînement du modèle
    print("Entraînement du modèle...")
    train_model(data_path=cleaned_tweets_file, epochs=5)

    # Étape 4 : Analyse des sentiments
    print("Analyse des sentiments...")
    analyze_sentiments(input_csv=cleaned_tweets_file)

    # Étape 5 : Lancement du dashboard
    print("Lancement du dashboard...")
    launch_dashboard()

if __name__ == "__main__":
    # Vérifier les arguments de ligne de commande
    source = 'api'  # Par défaut
    if len(sys.argv) > 1:
        source = sys.argv[1].lower()
    main(source=source)
