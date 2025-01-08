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
)

pie_chart = px.pie(
    sentiment_counts,
    names='sentiment',
    values='count',
    title="Répartition des Sentiments (Pie Chart)",
    color='sentiment'
)

# Application Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard des Sentiments", style={'textAlign': 'center'}),

    # Dropdown pour filtrer par sentiment
    html.Div([
        html.Label("Filtrer par Sentiment :"),
        dcc.Dropdown(
            id='sentiment-filter',
            options=[
                {'label': 'Tous', 'value': 'all'},
                {'label': 'Positif', 'value': 'Positive'},
                {'label': 'Négatif', 'value': 'Negative'},
                {'label': 'Neutre', 'value': 'Neutral'},
            ],
            value='all',  # Valeur par défaut
            clearable=False
        )
    ], style={'marginBottom': '20px', 'width': '50%'}),

    # Tableau
    html.Div([
        dash_table.DataTable(
            id='sentiment-table',
            columns=[{"name": col, "id": col} for col in df.columns],
            data=df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        )
    ], style={'marginBottom': '30px'}),

    # Graphiques
    html.Div([
        dcc.Graph(id='bar-chart', figure=bar_chart),
        dcc.Graph(id='pie-chart', figure=pie_chart),
    ], style={'display': 'flex', 'gap': '30px'})
])

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
    )

    updated_pie_chart = px.pie(
        sentiment_counts,
        names='sentiment',
        values='count',
        title="Répartition des Sentiments (Pie Chart)",
        color='sentiment',
    )

    return table_data, updated_bar_chart, updated_pie_chart


if __name__ == '__main__':
    app.run_server(debug=True)
