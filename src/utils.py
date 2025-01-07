# src/utils.py

import re
import string
import pandas as pd
from nltk.corpus import stopwords
import nltk

# Télécharger les stopwords (si nécessaire)
nltk.download('stopwords')

# Fonction pour nettoyer le texte d'un tweet
def clean_text(text):
    """
    Nettoie le texte d'un tweet en supprimant les mentions, hashtags, URL, ponctuation
    et stopwords (si nécessaire).
    
    :param text: Le texte du tweet à nettoyer.
    :return: Le texte nettoyé.
    """
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les mentions (ex: @username)
    text = re.sub(r'@[\w]+', '', text)
    
    # Supprimer les hashtags (si nécessaire)
    text = re.sub(r'#\w+', '', text)
    
    # Supprimer les URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer les stopwords (mots fréquents, non pertinents)
    stop_words = set(stopwords.words('french'))  # Utiliser la langue des tweets (ici, le français)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Fonction pour exporter les données vers un fichier CSV
def export_to_csv(data, filename):
    """
    Exporte les données sous forme de DataFrame pandas dans un fichier CSV.
    
    :param data: Les données à exporter (list of dicts).
    :param filename: Le nom du fichier CSV de sortie.
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Données exportées vers {filename}")
