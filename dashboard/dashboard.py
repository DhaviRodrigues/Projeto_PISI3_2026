import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
import dash
import dash_bootstrap_components as dbc
import eda
import xgboost_page
import cluster_page  

df = pd.read_parquet('Imdb_Movie_Dataset.parquet')

total_linhas = f"{df.shape[0]:,}".replace(",", ".")
total_colunas = df.shape[1]
tamanho_memoria = f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
}

CONTENT_STYLE = {
    "marginLeft": "9rem",
    "marginRight": "2rem",
}

sidebar = html.Div(
    [
        html.H2("Menu", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Página Inicial", href="/", active="exact"),
                dbc.NavLink("Exploratória (EDA)", href="/eda", active="exact"),
                dbc.NavLink("Clusterização", href="/cluster", active="exact"), 
                dbc.NavLink("Regressão", href="/regression", active="exact", disabled=True),
                dbc.NavLink("XGBoost", href="/xgboost_page", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],  
    style=SIDEBAR_STYLE,
)

content_home = html.Div([
    html.H1("Visão Geral do Dataset IMDB", className="mb-4"),
    
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total de Linhas"), html.P(total_linhas)]), color="info", outline=True)),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total de Colunas"), html.P(total_colunas)]), color="info", outline=True)),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Uso em Memória"), html.P(tamanho_memoria)]), color="info", outline=True)),
    ], className="mb-4"),

    html.H3("Dicionário de Colunas"),
    html.P("Abaixo estão listadas todas as colunas disponíveis e exemplos de dados:"),
    
    dash_table.DataTable(
        data=df.head(5).to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'minWidth': '150px', 'textAlign': 'left'},
        page_size=5
    ),
    
    html.Div([
        html.H4("Significado das Colunas Principais", className="mt-4"),
        html.Ul([
            html.Li([html.B("id:"), " Identificador único de cada filme no banco de dados."]),
            html.Li([html.B("title:"), " Título oficial do filme em seu lançamento comercial."]),
            html.Li([html.B("vote_average:"), " Nota média dada pelos usuários (0-10)."]),
            html.Li([html.B("vote_count:"), " Número total de votos e avaliações recebidas pelo filme."]),
            html.Li([html.B("status:"), " Estado atual da produção (ex: Released, Post Production, Rumored)."]),
            html.Li([html.B("release_date:"), " Data oficial de lançamento do filme."]),
            html.Li([html.B("revenue:"), " Receita bruta gerada pelo filme em dólares."]),
            html.Li([html.B("runtime:"), " Duração total do filme em minutos."]),
            html.Li([html.B("adult:"), " Indicador booleano (True/False) para classificação indicativa estritamente adulta."]),
            html.Li([html.B("budget:"), " Orçamento estimado de produção do filme em dólares."]),
            html.Li([html.B("original_language:"), " Idioma original em que o filme foi gravado (código ISO, ex: 'en', 'es', 'pt')."]),
            html.Li([html.B("original_title:"), " Título original do filme na língua nativa da produção."]),
            html.Li([html.B("overview:"), " Sinopse ou resumo do enredo do filme."]),
            html.Li([html.B("popularity:"), " Índice numérico de popularidade calculado pelo algoritmo da plataforma."]),
            html.Li([html.B("tagline:"), " Slogan, frase de efeito ou linha de chamada promocional do filme."]),
            html.Li([html.B("genres:"), " Lista de gêneros associados ao filme separados por vírgula (ex: Action, Drama)."]),
            html.Li([html.B("production_companies:"), " Companhias e estúdios responsáveis pela produção da obra."]),
            html.Li([html.B("production_countries:"), " Países onde o filme foi produzido ou financiado."]),
            html.Li([html.B("spoken_languages:"), " Idiomas falados e dublados disponíveis no corte do filme."]),
            html.Li([html.B("keywords:"), " Palavras-chave e tags de metadados que descrevem temas do filme."]),
        ])
    ], className="mt-4")

], style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, html.Div(id="page-content", style=CONTENT_STYLE) ])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return content_home
    elif pathname == "/eda":
        return eda.create_eda_layout(df)
    elif pathname == "/xgboost_page":
        return xgboost_page.create_xgboost_layout()
    elif pathname == "/cluster":
        return cluster_page.create_cluster_layout()
    
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"O caminho {pathname} não foi reconhecido."),
        ],
        className="p-3 bg-light rounded-3",
    )

eda.register_eda_callbacks(app, df)
xgboost_page.register_xgboost_callbacks(app)
cluster_page.register_cluster_callbacks(app)
    
if __name__ == '__main__':
    app.run(debug=True)