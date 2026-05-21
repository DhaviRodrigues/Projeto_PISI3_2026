import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'clusterizacao' / 'models'
DATA_PATH = BASE_DIR / 'Imdb_Movie_Dataset.parquet'

nomes_clusters = {
    0: "3 - Grandes Blockbusters",
    1: "1 - Baixo Consumo", 
    2: "2 - Aclamados pelo Público",
    3: "0 - Filmes Casuais" 
}

try:
    model_kmeans = joblib.load(MODELS_DIR / 'kmeans_model.pkl')
    model_scaler = joblib.load(MODELS_DIR / 'scaler_cluster.pkl')
    
    df_raw = pd.read_parquet(DATA_PATH)
    
    df_raw['release_year'] = pd.to_datetime(df_raw['release_date'], errors='coerce').dt.year
    
    features = ['popularity', 'vote_average', 'vote_count', 'runtime', 'budget', 'release_year']
    df_clustering = df_raw[features].copy()

    df_clustering = df_clustering[df_clustering['vote_count'] >= 50]
    
    cols_to_fix = ['runtime', 'budget', 'release_year']
    for col in cols_to_fix:
        df_clustering[col] = df_clustering[col].replace(0, np.nan)
        df_clustering[col] = df_clustering[col].fillna(df_clustering[col].median())
    
    df_clustering['popularity'] = np.log1p(df_clustering['popularity'])
    df_clustering['vote_count'] = np.log1p(df_clustering['vote_count'])
    df_clustering['budget'] = np.log1p(df_clustering['budget'])
    
    X_scaled = model_scaler.transform(df_clustering)
    
    df_result = df_raw.loc[df_clustering.index].copy()
    df_result['Cluster_ID'] = model_kmeans.predict(X_scaled)
    df_result['Cluster'] = df_result['Cluster_ID'].map(nomes_clusters)

except Exception as e:
    print(f"Erro ao sincronizar dados com o modelo: {e}")


def create_cluster_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Perfil dos Clusters (Médias Reais)", className="text-primary"),
                html.P("Análise visual sincronizada com o modelo (Filtro: >= 50 votos)."),
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Comparativo de Perfis (Clique na legenda para isolar/ocultar)"),
                    dbc.CardBody(dcc.Graph(id="cluster-parallel-plot"))
                ], className="shadow-sm mb-4")
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Resumo Estatístico Original"),
                    dbc.CardBody(id="tabela-medias-cluster")
                ], className="shadow-sm h-100")
            ], width=5),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Explorador de Filmes por Cluster ",
                        dcc.Dropdown(
                            id='dropdown-cluster-selector',
                            options=[{'label': v, 'value': v} for v in nomes_clusters.values()],
                            value="3 - Grandes Blockbusters",
                            clearable=False,
                            style={"width": "280px", "float": "right", "color": "black"}
                        )
                    ]),
                    dbc.CardBody(id="tabela-filmes-filtrados")
                ], className="shadow-sm h-100")
            ], width=7)
        ])
    ], style={"marginLeft": "18rem", "marginRight": "2rem", "padding": "2rem"})

def register_cluster_callbacks(app):
    
    @app.callback(
        [Output("cluster-parallel-plot", "figure"),
         Output("tabela-medias-cluster", "children")],
        [Input("dropdown-cluster-selector", "value")]
    )
    def update_charts(_):
        colunas_analise = ['popularity', 'vote_average', 'vote_count', 'budget', 'runtime', 'release_year']
        df_perfil = df_result.groupby('Cluster')[colunas_analise].mean().reset_index()
        
        df_perfil.columns = ['Cluster', 'Popularidade', 'Nota Média', 'Total Votos', 'Orçamento', 'Duração (min)', 'Ano Lançamento']
        df_perfil = df_perfil.sort_values('Cluster')

        fig = go.Figure()
        metricas_plot = ['Popularidade', 'Nota Média', 'Total Votos', 'Orçamento', 'Duração (min)']
        
        df_plot = df_perfil.copy()
        for col in metricas_plot:
            max_val = df_plot[col].max()
            df_plot[f'{col}_norm'] = df_plot[col] / max_val if max_val > 0 else 0

        cores = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        
        for i, row in df_plot.reset_index(drop=True).iterrows():
            valores_reais = [row[col] for col in metricas_plot]
            valores_norm = [row[f'{col}_norm'] for col in metricas_plot]
            
            hover_text = [f"<b>{col}</b>: {val:,.2f}" for col, val in zip(metricas_plot, valores_reais)]
            
            fig.add_trace(go.Scatter(
                x=metricas_plot,
                y=valores_norm,
                mode='lines+markers',
                name=row['Cluster'], 
                line=dict(width=4, color=cores[i % len(cores)]),
                marker=dict(size=10),
                text=hover_text,
                hoverinfo="name+text"
            ))

        fig.update_layout(
            yaxis=dict(showticklabels=False, title="Escala Proporcional ao Máximo"),
            margin=dict(t=20, b=30),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        df_tabela_show = df_perfil[['Cluster', 'Popularidade', 'Nota Média', 'Total Votos', 'Orçamento']].copy()
        df_tabela_show['Orçamento'] = df_tabela_show['Orçamento'].apply(lambda x: f"${x/1e6:.1f}M" if x > 0 else "$0.0M")
        df_tabela_show['Popularidade'] = df_tabela_show['Popularidade'].round(1)
        df_tabela_show['Nota Média'] = df_tabela_show['Nota Média'].round(2)
        df_tabela_show['Total Votos'] = df_tabela_show['Total Votos'].round(0)

        tabela = dbc.Table.from_dataframe(
            df_tabela_show, 
            striped=True, bordered=True, hover=True
        )
        
        return fig, tabela

    @app.callback(
        Output("tabela-filmes-filtrados", "children"),
        Input("dropdown-cluster-selector", "value")
    )
    def update_movie_table(cluster_selecionado):
        filmes = df_result[df_result['Cluster'] == cluster_selecionado].sort_values('popularity', ascending=False).head(50)
        
        df_disp = filmes[['title', 'popularity', 'vote_average', 'vote_count', 'runtime']].copy()
        df_disp.columns = ['Título', 'Popularidade', 'Nota Média', 'Votos', 'Duração (min)']
        
        return dash_table.DataTable(
            data=df_disp.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df_disp.columns],
            page_size=10,
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
        )