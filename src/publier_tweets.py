import tweepy
from twitter_auth import authenticate_twitter_api  # Importer votre fonction d'authentification

# Charger les tweets depuis le fichier .txt
def read_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tweets = f.readlines()
    return [tweet.strip() for tweet in tweets]

# Publier un tweet
def publish_tweet(api, tweet):
    try:
        api.update_status(tweet)  # Publie un tweet sur Twitter
        print(f"Tweet publié : {tweet}")
    except Exception as e:
        print(f"Erreur lors de la publication : {e}")

# Fonction principale pour publier tous les tweets
def main():
    # Authentification avec l'API Twitter
    api = authenticate_twitter_api()

    # Lire les tweets depuis le fichier "tweets.txt"
    tweets = read_tweets("notebooks/tweets.txt")  # Le fichier est dans le répertoire notebooks

    # Publier chaque tweet
    for tweet in tweets:
        publish_tweet(api, tweet)

if __name__ == "__main__":
    main()
