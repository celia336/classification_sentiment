import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Charger les résultats des sentiments (par exemple, le fichier CSV exporté)
df = pd.read_csv('data/processed/tweets_sentiment.csv')  # Assurez-vous que le chemin est correct

# Créer un Dash app
app = dash.Dash(__name__)

# Créer un graphique en camembert pour afficher la répartition des sentiments
sentiment_counts = df['sentiment'].value_counts()
fig = px.pie(values=sentiment_counts, names=sentiment_counts.index, title="Répartition des sentiments")

# Créer un graphique de la tendance des sentiments au fil du temps
df['created_at'] = pd.to_datetime(df['created_at'])
df['date'] = df['created_at'].dt.date
sentiment_by_date = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
fig2 = px.line(sentiment_by_date, x=sentiment_by_date.index, y=sentiment_by_date.columns, title="Évolution des sentiments au fil du temps")

# Layout du dashboard avec deux graphiques
app.layout = html.Div([
    html.H1("Dashboard d'analyse des sentiments des tweets"),
    
    # Premier graphique : Répartition des sentiments
    dcc.Graph(
        id='sentiment-pie-chart',
        figure=fig
    ),
    
    # Deuxième graphique : Évolution des sentiments au fil du temps
    dcc.Graph(
        id='sentiment-time-line',
        figure=fig2
    )
])

# Fonction pour lancer le dashboard
def launch_dashboard():
    app.run_server(debug=True)

# Lancer l'app si ce fichier est exécuté directement
if __name__ == '__main__':
    launch_dashboard()
