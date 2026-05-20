import pandas as pd
import joblib
import os
from pathlib import Path
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'clusterizacao' / 'models'
DATA_PATH = BASE_DIR / 'Imdb_Movie_Dataset.parquet'

model_kmeans = None
model_scaler = None
df_clustered = None
features = ['popularity', 'vote_average', 'vote_count', 'runtime']

try:
    model_kmeans = joblib.load(MODELS_DIR / 'kmeans_model.pkl')
    model_scaler = joblib.load(MODELS_DIR / 'scaler_cluster.pkl')
    
    df = pd.read_parquet(DATA_PATH)
    df_clustered = df.copy()
    
    df_clustered['runtime'] = df_clustered['runtime'].replace(0, pd.NA)
    df_clustered = df_clustered.dropna(subset=features)
    
    X_scaled = model_scaler.transform(df_clustered[features])
    df_clustered['Cluster'] = model_kmeans.predict(X_scaled)
    
    nomes_clusters = {
        0: "0 - Filmes Casuais",
        1: "1 - Baixo Consumo", 
        2: "2 - Aclamados pelo Público",
        3: "3 - Grandes Blockbusters"
    }
    df_clustered['Cluster'] = df_clustered['Cluster'].map(nomes_clusters)

except Exception as e:
    print(f"Erro ao carregar arquivos de Clusterização: {e}")

def create_cluster_layout():
    """Gera o layout da página de Clusterização, seguindo o padrão do XGBoost"""
    
    if df_clustered is None or model_kmeans is None:
        return html.Div(
            dbc.Alert("❌ Erro ao carregar os modelos ou o dataset. Verifique a pasta 'models' e o arquivo parquet.", color="danger"),
            style={"marginLeft": "18rem", "marginRight": "2rem", "paddingTop": "1rem"}
        )

    opcoes_clusters = sorted(df_clustered['Cluster'].unique())

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Agrupamento de Filmes (K-Means)", className="mb-2"),
                html.P("Descubra os perfis de filmes agrupados por popularidade, avaliação, contagem de votos e duração.", className="text-muted"),
            ], width=12),
        ], className="mb-4 align-items-end"),
        html.Hr(className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Distribuição dos Clusters", className="m-0")),
                    dbc.CardBody([
                        dcc.Loading(
                            type="default",
                            children=dcc.Graph(id="cluster-scatter-plot")
                        )
                    ])
                ], outline=True, color="primary", className="shadow-sm h-100")
            ], width=7),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Perfil dos Clusters (Médias)", className="m-0")),
                    dbc.CardBody([
                         html.Div(id="tabela-medias-cluster")
                    ])
                ], outline=True, color="primary", className="shadow-sm h-100")
            ], width=5)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("Explorador de Filmes", className="m-0 d-inline-block align-middle"),
                            html.Div([
                                html.Label("Filtrar por Perfil:", className="me-2 fw-bold align-middle"),
                                dcc.Dropdown(
                                    id='dropdown-cluster-selector',
                                    options=[{'label': c, 'value': c} for c in opcoes_clusters],
                                    value=opcoes_clusters[0], # Inicia com cluster 0
                                    multi=False,
                                    clearable=False,
                                    style={"width": "320px"} # Aumentado para caber o novo nome
                                )
                            ], className="float-end d-flex align-items-center", style={"position": "relative", "zIndex": 9999})
                        ], className="w-100")
                    ]),
                    dbc.CardBody([
                        html.Div(id="tabela-filmes-filtrados")
                    ])
                ], outline=True, color="primary", className="shadow-sm")
            ], width=12)
        ], className="mb-4")
        
    ], style={"marginLeft": "18rem", "marginRight": "2rem", "paddingTop": "1rem"}) # Margem padrão do seu projeto

def register_cluster_callbacks(app):
    """Registra os callbacks para tornar a página interativa"""
    
    if df_clustered is None:
        return

    @app.callback(
        [Output("cluster-scatter-plot", "figure"),
         Output("tabela-medias-cluster", "children")],
        Input("dropdown-cluster-selector", "value")
    )
    def update_static_visuals(_):
        fig = px.scatter(
            df_clustered,
            x='popularity',
            y='vote_average',
            color='Cluster',
            hover_data=['title', 'vote_count', 'runtime'],
            color_discrete_sequence=px.colors.qualitative.Pastel,
            opacity=0.7,
            labels={'popularity': 'Popularidade', 'vote_average': 'Nota Média', 'Cluster': 'Perfil'}
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20), 
            paper_bgcolor="rgba(0,0,0,0)",
            legend_itemsizing="constant"
        )

        perfil = df_clustered.groupby('Cluster')[features].mean().reset_index()
        perfil.columns = ['Cluster', 'Popularidade', 'Nota Média', 'Votos', 'Duração (min)']
        
        perfil['Popularidade'] = perfil['Popularidade'].astype(float).round(2)
        perfil['Nota Média'] = perfil['Nota Média'].astype(float).round(2)
        perfil['Votos'] = perfil['Votos'].astype(float).round(0)
        perfil['Duração (min)'] = perfil['Duração (min)'].astype(float).round(2)
        
        tabela = dbc.Table.from_dataframe(perfil, striped=True, bordered=True, hover=True, size="sm")
        
        return fig, tabela

    @app.callback(
        Output("tabela-filmes-filtrados", "children"),
        Input("dropdown-cluster-selector", "value")
    )
    def update_movie_table(cluster_selecionado):
        if not cluster_selecionado:
            return html.Div("Selecione um perfil.")

        if isinstance(cluster_selecionado, str):
            cluster_selecionado = [cluster_selecionado]

        filmes = df_clustered[df_clustered['Cluster'].isin(cluster_selecionado)]
        
        filmes_display = filmes[['title', 'popularity', 'vote_average', 'vote_count', 'runtime']].head(100)
        filmes_display.columns = ['Título', 'Popularidade', 'Nota Média', 'Votos', 'Duração (min)']
        
        filmes_display['Popularidade'] = filmes_display['Popularidade'].astype(float).round(2)
        filmes_display['Nota Média'] = filmes_display['Nota Média'].astype(float).round(1)
        filmes_display['Duração (min)'] = filmes_display['Duração (min)'].astype(float).round(0)

        table = dash_table.DataTable(
            data=filmes_display.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in filmes_display.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
        )
        
        return table