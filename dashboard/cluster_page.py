import pandas as pd
from pathlib import Path
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'clusterizacao' / 'models' / 'Imdb_Movie_Dataset_Clustered.parquet'

CLUSTER_NAMES = {
    0: "Cluster 0 - Grandes Blockbusters",
    1: "Cluster 1 - Mercado Intermediário",
    2: "Cluster 2 - Cinema Independente",
    3: "Cluster 3 - Baixo Orçamento"
}

try:
    df_result = pd.read_parquet(DATA_PATH)

    cols_numericas = ['popularity', 'vote_average', 'vote_count', 'budget', 'runtime', 'release_year']
    for col in cols_numericas:
        if col in df_result.columns:
            df_result[col] = pd.to_numeric(df_result[col], errors='coerce')

    clusters_unicos = sorted(df_result['Cluster'].unique().tolist()) if not df_result.empty else []
except Exception as e:
    print(f"Erro ao carregar o ficheiro Parquet do cluster: {e}")
    df_result = pd.DataFrame()
    clusters_unicos = []


def create_cluster_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Perfil dos Clusters", className="text-primary"),
                html.P("Análise visual sincronizada com o modelo (Filtro: >= 5 votos)."),
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
                ], className="shadow-sm mb-4")
            ], width=12)
        ]),

        dbc.Row([  
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Explorador de Filmes por Cluster ",
                        dcc.Dropdown(
                            id='dropdown-cluster-selector',
                            options=[{'label': CLUSTER_NAMES.get(v, f"Cluster {v}"), 'value': v} for v in clusters_unicos],
                            value=clusters_unicos[0] if clusters_unicos else None,
                            clearable=False,
                            style={"width": "350px", "float": "right", "color": "black"}
                        )
                    ]),
                    dbc.CardBody(id="tabela-filmes-filtrados")
                ], className="shadow-sm mb-4")
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sistema de Recomendação Híbrido (K-Means e DBSCAN)", className="fw-bold bg-primary text-white"),
                    dbc.CardBody([
                        html.P("Digite o nome exato de um filme (em inglês) para buscar obras similares:"),
                        dbc.Row([
                            dbc.Col(
                                dbc.Input(id="input-movie-title", type="text", placeholder="Ex: The Matrix, Inception, Titanic..."), 
                                width=9
                            ),
                            dbc.Col(
                                dbc.Button("Recomendar", id="btn-recommend", color="success", className="w-100"), 
                                width=3
                            )
                        ]),
                        html.Hr(),
                        html.Div(id="recommendation-output", className="mt-3")
                    ])
                ], className="shadow-sm mb-4")
            ], width=12)
        ])
    ], style={"marginLeft": "18rem", "marginRight": "2rem", "padding": "2rem"})


def register_cluster_callbacks(app):
    
    @app.callback(
        [Output("cluster-parallel-plot", "figure"),
         Output("tabela-medias-cluster", "children")],
        [Input("dropdown-cluster-selector", "value")]
    )
    def update_charts(_):
        if df_result.empty:
            return go.Figure(), html.P("Sem dados disponíveis.")

        colunas_analise = ['popularity', 'vote_average', 'vote_count', 'budget', 'runtime', 'release_year']
        df_perfil = df_result.groupby('Cluster')[colunas_analise].mean().reset_index()
        df_perfil.columns = ['Cluster', 'Popularidade', 'Nota Média', 'Média de Votos', 'Orçamento', 'Duração (min)', 'Ano Lançamento']
        df_perfil = df_perfil.sort_values('Cluster')
        
        df_tabela_show = df_perfil[['Cluster', 'Popularidade', 'Nota Média', 'Média de Votos', 'Orçamento']].copy()
        df_tabela_show['Cluster'] = df_tabela_show['Cluster'].map(CLUSTER_NAMES)
        df_tabela_show['Orçamento'] = df_tabela_show['Orçamento'].apply(lambda x: f"${x/1e6:.1f}M" if x > 0 else "$0.0M")
        df_tabela_show['Popularidade'] = df_tabela_show['Popularidade'].round(1)
        df_tabela_show['Nota Média'] = df_tabela_show['Nota Média'].round(2)
        df_tabela_show['Média de Votos'] = df_tabela_show['Média de Votos'].round(0)

        tabela = dbc.Table.from_dataframe(df_tabela_show, striped=True, bordered=True, hover=True)

        sample_size_per_cluster = 700
        sampled_indices = []
        for cluster_id in df_result['Cluster'].unique():
            cluster_df = df_result[df_result['Cluster'] == cluster_id]
            sampled_indices.extend(cluster_df.sample(min(len(cluster_df), sample_size_per_cluster), random_state=42).index.tolist())

        sampled_df = df_result.loc[sampled_indices].reset_index(drop=True)
        

        sampled_df['Cluster_Cat'] = sampled_df['Cluster'].map(CLUSTER_NAMES)

        fig = px.scatter(
            sampled_df,
            x='UMAP_1',
            y='UMAP_2',
            color='Cluster_Cat',
            hover_name='title',
            hover_data=['Cluster', 'vote_average', 'popularity', 'vote_count', 'genres'],
            labels={
                'UMAP_1': 'Componente UMAP 1', 
                'UMAP_2': 'Componente UMAP 2',
                'Cluster_Cat': 'Perfil de Mercado'
            },
            color_discrete_sequence=px.colors.qualitative.G10
        )

        fig.update_layout(
            margin=dict(t=20, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor="white"
        )
    
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        return fig, tabela

    @app.callback(
        Output("tabela-filmes-filtrados", "children"),
        Input("dropdown-cluster-selector", "value")
    )
    def update_movie_table(cluster_selecionado):
        if df_result.empty or cluster_selecionado is None:
            return html.P("Sem dados disponíveis.")

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
    
    @app.callback(
        Output("recommendation-output", "children"),
        Input("btn-recommend", "n_clicks"),
        State("input-movie-title", "value")
    )
    def generate_recommendations(n_clicks, titulo):
        if not n_clicks or not titulo:
            return html.Div()

        filme_exato = df_result[df_result['title'].str.lower() == titulo.lower()]
        
        if filme_exato.empty:
            return html.Div(f"Não encontramos o filme '{titulo}'. Verifique a ortografia (nome em inglês).", className="text-danger fw-bold")
        
        filme_alvo = filme_exato.iloc[0]
        nome_oficial = filme_alvo['title']

        x_alvo = filme_alvo['UMAP_1']
        y_alvo = filme_alvo['UMAP_2']
        
        cluster_k = filme_alvo['Cluster']
        cluster_db = filme_alvo['cluster_dbscan']
        
        recomendacoes_finais = []
        origens = []

        def buscar_vizinhos(df_filtrado, limite):
            distancias = np.sqrt((df_filtrado['UMAP_1'] - x_alvo)**2 + (df_filtrado['UMAP_2'] - y_alvo)**2)
            df_temp = df_filtrado.copy()
            df_temp['distancia_real'] = distancias
            return df_temp.sort_values('distancia_real', ascending=True).head(limite)['title'].tolist()

        if cluster_db != -1:
            df_dbscan = df_result[(df_result['cluster_dbscan'] == cluster_db) & (df_result['title'] != nome_oficial)]
            recs_dbscan = buscar_vizinhos(df_dbscan, 5)
            recomendacoes_finais.extend(recs_dbscan)
            origens.extend(["DBSCAN"] * len(recs_dbscan))

        vagas_sobrando = 5 - len(recomendacoes_finais)
        
        if vagas_sobrando > 0:
            df_kmeans = df_result[
                (df_result['Cluster'] == cluster_k) & 
                (df_result['title'] != nome_oficial) & 
                (~df_result['title'].isin(recomendacoes_finais))
            ]
            recs_kmeans = buscar_vizinhos(df_kmeans, vagas_sobrando)
            recomendacoes_finais.extend(recs_kmeans)
            origens.extend(["K-Means"] * len(recs_kmeans))

        if not recomendacoes_finais:
            return html.Div("Nenhum filme similar encontrado.", className="text-warning")

        list_items = []
        for filme, origem in zip(recomendacoes_finais, origens):
            badge_color = "success" if origem == "DBSCAN" else "info"
            list_items.append(
                dbc.ListGroupItem([
                    html.Span(filme, className="fw-bold"),
                    dbc.Badge(f"via {origem}", color=badge_color, className="ms-2")
                ], className="d-flex justify-content-between align-items-center")
            )
            
        return html.Div([
            html.H5(f"Recomendações por Similaridade Real para: {nome_oficial}", className="text-primary mb-3 fw-bold"),
            dbc.ListGroup(list_items, className="shadow-sm")
        ])