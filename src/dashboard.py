import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import plotly.express as px

# Chargement des données depuis le fichier CSV
csv_file = "data/processed/tweets_sentiment.csv"  # Remplace par le nom de ton fichier CSV
df = pd.read_csv(csv_file)

# Vérification des colonnes nécessaires
if 'text' not in df.columns or 'sentiment' not in df.columns:
    raise ValueError("Le fichier CSV doit contenir les colonnes 'text' et 'sentiment'.")

# Calcul de la répartition des sentiments
sentiment_counts = df['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']  # Renomme les colonnes

# Création des graphiques
bar_chart = px.bar(
    sentiment_counts,
    x='sentiment',
    y='count',
    labels={'sentiment': 'Sentiment', 'count': 'Nombre'},
    title="Répartition des Sentiments (Bar Chart)",
    color='sentiment',
    color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"},
)

pie_chart = px.pie(
    sentiment_counts,
    names='sentiment',
    values='count',
    title="Répartition des Sentiments (Pie Chart)",
    color='sentiment',
    color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"},
)

# Application Dash
app = dash.Dash(__name__)

app.css.config.serve_locally = True
app.title = "Sentiment Dashboard"

# Mise en page de l'application
app.layout = html.Div([
    html.Div([
        # Barre latérale pour les filtres
        html.Div([
            html.H2("Contrôles", style={'color': '#2c3e50'}),
            html.Label("Filtrer par Sentiment :", style={'marginTop': '20px', 'color': '#34495e'}),
            dcc.Dropdown(
                id='sentiment-filter',
                options=[
                    {'label': 'Tous', 'value': 'all'},
                    {'label': 'Positif', 'value': 'Positive'},
                    {'label': 'Négatif', 'value': 'Negative'},
                    {'label': 'Neutre', 'value': 'Neutral'},
                ],
                value='all',  # Valeur par défaut
                clearable=False,
                style={'width': '100%', 'marginBottom': '20px'}
            )
        ], style={
            'padding': '20px',
            'backgroundColor': '#f7dc6f',  # Jaune clair pour la barre latérale
            'borderRadius': '10px',
            'boxShadow': '0px 2px 5px rgba(0,0,0,0.1)',
            'flex': '1'
        }),

        # Contenu principal
        html.Div([
            html.H1("Dashboard des Sentiments", style={
                'textAlign': 'center', 
                'color': '#2c3e50', 
                'marginBottom': '30px'
            }),

            # Tableau des textes
            html.Div([
                html.H4("Tableau des Textes", style={'color': '#34495e'}),
                dash_table.DataTable(
                    id='sentiment-table',
                    columns=[{"name": col, "id": col} for col in df.columns],
                    data=df.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                )
            ], style={
                'marginBottom': '30px',
                'backgroundColor': '#fef9e7',  # Jaune clair pour le tableau
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0px 2px 5px rgba(0,0,0,0.1)'
            }),

            # Graphiques
            html.Div([
                dcc.Graph(id='bar-chart', figure=bar_chart, style={'flex': '1'}),
                dcc.Graph(id='pie-chart', figure=pie_chart, style={'flex': '1'}),
            ], style={
                'display': 'flex',
                'gap': '20px',
                'justifyContent': 'space-between',
            }),
        ], style={'flex': '3'})
    ], style={'display': 'flex', 'gap': '20px', 'padding': '20px'})
], style={
    'backgroundColor': '#f9e79f',  # Fond jaune global
    'padding': '20px',
    'minHeight': '100vh'  # Prend toute la hauteur de la fenêtre
})

# Callback pour mettre à jour le tableau et les graphiques en fonction du filtre
@app.callback(
    [Output('sentiment-table', 'data'),
     Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('sentiment-filter', 'value')]
)
def update_dashboard(selected_sentiment):
    if selected_sentiment == 'all':
        filtered_df = df
    else:
        filtered_df = df[df['sentiment'] == selected_sentiment]

    # Mettre à jour le tableau
    table_data = filtered_df.to_dict('records')

    # Mettre à jour les graphiques
    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']

    updated_bar_chart = px.bar(
        sentiment_counts,
        x='sentiment',
        y='count',
        labels={'sentiment': 'Sentiment', 'count': 'Nombre'},
        title="Répartition des Sentiments (Bar Chart)",
        color='sentiment',
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"},
    )

    updated_pie_chart = px.pie(
        sentiment_counts,
        names='sentiment',
        values='count',
        title="Répartition des Sentiments (Pie Chart)",
        color='sentiment',
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"},
    )

    return table_data, updated_bar_chart, updated_pie_chart


if __name__ == '__main__':
    app.run_server(debug=True)
