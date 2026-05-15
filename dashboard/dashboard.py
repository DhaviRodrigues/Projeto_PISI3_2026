import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
import dash
import dash_bootstrap_components as dbc
import eda

df = pd.read_parquet('Projeto_PISI3_2026\Imdb_Movie_Dataset.parquet')

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
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Menu", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Página Inicial", href="/", active="exact"),
                dbc.NavLink("Exploratória (EDA)", href="/eda", active="exact"),
                dbc.NavLink("Clusters", href="/cluster", active="exact", disabled=True),
                dbc.NavLink("Random Forest", href="/random_forest", active="exact", disabled=True),
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
            html.Li([html.B("revenue:"), " Receita bruta gerada pelo filme em dólares."]),
            html.Li([html.B("vote_average:"), " Nota média dada pelos usuários (0-10)."]),
            html.Li([html.B("popularity:"), " Índice numérico de popularidade calculado pelo algoritmo da plataforma."]),
            html.Li([html.B("runtime:"), " Duração do filme em minutos."]),
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
    
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"O caminho {pathname} não foi reconhecido."),
        ],
        className="p-3 bg-light rounded-3",
    )

eda.register_eda_callbacks(app, df)

if __name__ == '__main__':
    app.run(debug=True)