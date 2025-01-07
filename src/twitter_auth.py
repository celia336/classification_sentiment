import tweepy
import os  # Importer le module os pour récupérer les variables d'environnement

def authenticate_twitter_api():
    # Récupérer les clés et tokens depuis les variables d'environnement
    consumer_key = os.getenv('CONSUMER_KEY')  # Récupérer la clé d'API consommateur
    consumer_secret = os.getenv('CONSUMER_SECRET')  # Récupérer le secret d'API consommateur
    access_token = os.getenv('ACCESS_TOKEN')  # Récupérer le token d'accès
    access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')  # Récupérer le secret du token d'accès
    
    # Vérifier que les clés et tokens sont correctement récupérés
    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        print("Erreur: Les clés ou tokens d'API ne sont pas définis.")
        return None
    
    # Authentification avec l'API Twitter                                                                                                    
    auth = tweepy.OAuth1UserHandler(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )
    
    # Retourner l'API authentifiée
    api = tweepy.API(auth)
    
    try:
        # Vérifier la connexion en demandant des informations sur l'utilisateur
        api.verify_credentials()
        print("Authentification réussie")
    except tweepy.TweepyException as e:
        print(f"Erreur d'authentification : {e}")
    
    return api
