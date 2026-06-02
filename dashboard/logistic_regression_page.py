import pandas as pd
import numpy as np
from pathlib import Path
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ─────────────────────────────────────────────
# CAMINHOS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'clusterizacao' / 'models' / 'Imdb_Movie_Dataset_Clustered.parquet'

# ─────────────────────────────────────────────
# CARREGAMENTO E TREINAMENTO (executado 1 vez no import)
# ─────────────────────────────────────────────
LIMIAR_SUCESSO = 7.0

def _carregar_e_treinar():
    """Carrega o dataset e treina o modelo de Regressão Logística."""
    df = pd.read_parquet(DATA_PATH)

    # ── Preparação dos dados de gênero ──
    df_ml = df.dropna(subset=['genres']).copy()
    df_ml['genre_list'] = df_ml['genres'].str.split(', ')

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df_ml['genre_list'])
    y = (df_ml['vote_average'] >= LIMIAR_SUCESSO).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=['Nota < 7', 'Nota >= 7'],
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    # ── Análise por gênero ──
    df_genres = df.dropna(subset=['genres']).copy()
    df_genres['genre_list'] = df_genres['genres'].str.split(', ')
    df_exploded = df_genres.explode('genre_list').dropna(subset=['genre_list'])
    df_exploded['is_success'] = (df_exploded['vote_average'] >= LIMIAR_SUCESSO).astype(int)

    genre_analysis = df_exploded.groupby('genre_list').agg(
        success_probability=('is_success', 'mean'),
        vote_average_mean=('vote_average', 'mean'),
        movie_count=('id', 'count')
    ).reset_index()
    genre_analysis = genre_analysis[genre_analysis['movie_count'] > 500].sort_values(
        by='success_probability', ascending=False
    )

    genre_counts = df_exploded['genre_list'].value_counts().head(15).reset_index()
    genre_counts.columns = ['genre', 'count']

    # Coeficientes por gênero
    coef_df = pd.DataFrame({
        'genre': mlb.classes_,
        'coef': model.coef_[0]
    }).sort_values('coef', ascending=False)

    return {
        'model': model,
        'mlb': mlb,
        'accuracy': accuracy,
        'report': report,
        'cm': cm,
        'genre_analysis': genre_analysis,
        'genre_counts': genre_counts,
        'coef_df': coef_df,
        'all_genres': sorted(mlb.classes_.tolist()),
    }

try:
    _DATA = _carregar_e_treinar()
except Exception as e:
    print(f"[logistic_regression_page] Erro ao carregar dados: {e}")
    _DATA = None


# ─────────────────────────────────────────────
# HELPERS DE FIGURAS
# ─────────────────────────────────────────────
_TEMPLATE = "plotly_white"
_PRIMARY   = "#2C7BE5"
_SUCCESS   = "#00D97E"
_DANGER    = "#E63757"
_SECONDARY = "#95AAC9"


def _fig_genre_frequency(genre_counts: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=genre_counts['count'],
        y=genre_counts['genre'],
        orientation='h',
        marker_color=_PRIMARY,
        hovertemplate='<b>%{y}</b><br>Filmes: %{x:,}<extra></extra>'
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title="Top 15 Gêneros Mais Frequentes",
        xaxis_title="Quantidade de Filmes",
        yaxis_title=None,
        margin=dict(l=130, r=20, t=50, b=40),
        height=420,
    )
    return fig


def _fig_success_prob(genre_analysis: pd.DataFrame) -> go.Figure:
    colors = [_SUCCESS if v >= 0.25 else _SECONDARY for v in genre_analysis['success_probability']]
    fig = go.Figure(go.Bar(
        x=genre_analysis['success_probability'] * 100,
        y=genre_analysis['genre_list'],
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>Probabilidade: %{x:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title=f"Probabilidade de Nota ≥ {LIMIAR_SUCESSO} por Gênero",
        xaxis_title="Probabilidade de Sucesso (%)",
        yaxis_title=None,
        margin=dict(l=130, r=20, t=50, b=40),
        height=420,
    )
    return fig


def _fig_coef(coef_df: pd.DataFrame) -> go.Figure:
    colors = [_SUCCESS if c > 0 else _DANGER for c in coef_df['coef']]
    fig = go.Figure(go.Bar(
        x=coef_df['coef'],
        y=coef_df['genre'],
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>Coeficiente: %{x:.4f}<extra></extra>'
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title="Coeficientes do Modelo (impacto de cada gênero)",
        xaxis_title="Coeficiente Logístico",
        yaxis_title=None,
        shapes=[dict(type='line', x0=0, x1=0, y0=-0.5,
                     y1=len(coef_df)-0.5, line=dict(color='black', width=1, dash='dot'))],
        margin=dict(l=130, r=20, t=50, b=40),
        height=500,
    )
    return fig


def _fig_confusion_matrix(cm: np.ndarray) -> go.Figure:
    labels = ['Nota < 7', 'Nota ≥ 7']
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        hovertemplate='Real: %{y}<br>Previsto: %{x}<br>Qtd: %{z}<extra></extra>'
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title="Matriz de Confusão",
        xaxis_title="Previsto",
        yaxis_title="Real",
        height=350,
        margin=dict(l=80, r=20, t=50, b=60),
    )
    return fig


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
def create_logistic_layout():
    if _DATA is None:
        return html.Div([
            dbc.Alert("Erro ao carregar os dados ou treinar o modelo. Verifique o caminho do dataset.", color="danger")
        ], style={"marginLeft": "18rem", "marginRight": "2rem", "paddingTop": "2rem"})

    d = _DATA
    accuracy_pct = f"{d['accuracy']:.2%}"

    # Cards de métricas rápidas
    report = d['report']
    cards_metricas = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Acurácia Geral", className="card-subtitle text-muted mb-1"),
            html.H3(accuracy_pct, className="text-primary mb-0")
        ]), className="shadow-sm"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Precisão (Nota ≥ 7)", className="card-subtitle text-muted mb-1"),
            html.H3(f"{report['Nota >= 7']['precision']:.2%}", className="text-success mb-0")
        ]), className="shadow-sm"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Recall (Nota ≥ 7)", className="card-subtitle text-muted mb-1"),
            html.H3(f"{report['Nota >= 7']['recall']:.2%}", className="text-warning mb-0")
        ]), className="shadow-sm"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("F1-Score (Nota ≥ 7)", className="card-subtitle text-muted mb-1"),
            html.H3(f"{report['Nota >= 7']['f1-score']:.2%}", className="text-info mb-0")
        ]), className="shadow-sm"), width=3),
    ], className="mb-4 g-3")

    # Simulador de predição
    genre_options = [{"label": g, "value": g} for g in d['all_genres']]
    simulador = dbc.Card([
        dbc.CardHeader(html.H5("🎬 Simulador de Sucesso", className="mb-0")),
        dbc.CardBody([
            html.P("Selecione os gêneros do filme e descubra a probabilidade de ele ter nota ≥ 7.0 no IMDB.",
                   className="text-muted"),
            dbc.Row([
                dbc.Col([
                    html.Label("Gêneros do Filme:", className="fw-bold"),
                    dcc.Dropdown(
                        id="lr-input-genres",
                        options=genre_options,
                        multi=True,
                        value=["Drama"],
                        placeholder="Selecione um ou mais gêneros...",
                        className="mb-3"
                    ),
                    dbc.Button(
                        "Calcular Probabilidade",
                        id="lr-btn-predict",
                        color="primary",
                        className="w-100",
                        n_clicks=0
                    ),
                ], width=6),
                dbc.Col([
                    html.Div(id="lr-resultado", className="h-100 d-flex align-items-center justify-content-center")
                ], width=6)
            ])
        ])
    ], className="shadow-sm mb-4")

    return html.Div([
        # Cabeçalho
        dbc.Row([
            dbc.Col([
                html.H1("Regressão Logística — Previsão de Sucesso", className="mb-1 mt-4"),
                html.P(
                    f"Modelo treinado para classificar filmes com nota ≥ {LIMIAR_SUCESSO} "
                    "usando apenas os gêneros como features.",
                    className="text-muted"
                )
            ])
        ], className="mb-3"),

        # Cards de métricas
        cards_metricas,

        # Simulador
        simulador,

        # Gráficos de análise
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Análise Exploratória & Modelo", className="mb-0")),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="Frequência de Gêneros", tab_id="tab-freq",
                                    children=[dcc.Graph(figure=_fig_genre_frequency(d['genre_counts']), id="lr-fig-freq")]),
                            dbc.Tab(label="Probabilidade de Sucesso", tab_id="tab-prob",
                                    children=[dcc.Graph(figure=_fig_success_prob(d['genre_analysis']), id="lr-fig-prob")]),
                            dbc.Tab(label="Coeficientes do Modelo", tab_id="tab-coef",
                                    children=[dcc.Graph(figure=_fig_coef(d['coef_df']), id="lr-fig-coef")]),
                            dbc.Tab(label="Matriz de Confusão", tab_id="tab-cm",
                                    children=[dcc.Graph(figure=_fig_confusion_matrix(d['cm']), id="lr-fig-cm")]),
                        ], active_tab="tab-freq")
                    ])
                ], className="shadow-sm")
            ])
        ]),
    ], style={"marginLeft": "18rem", "marginRight": "2rem", "paddingTop": "1rem"})


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
def register_logistic_callbacks(app):

    @app.callback(
        Output("lr-resultado", "children"),
        Input("lr-btn-predict", "n_clicks"),
        State("lr-input-genres", "value"),
        prevent_initial_call=True
    )
    def prever_sucesso(n_clicks, selected_genres):
        if not n_clicks or not selected_genres:
            return dbc.Alert("Selecione ao menos um gênero.", color="warning")

        if _DATA is None:
            return dbc.Alert("Modelo não disponível.", color="danger")

        try:
            model = _DATA['model']
            mlb   = _DATA['mlb']

            generos_encoded = mlb.transform([selected_genres])
            prob_sucesso = model.predict_proba(generos_encoded)[0][1]
            prob_falha   = 1 - prob_sucesso

            cor   = "success" if prob_sucesso >= 0.5 else "warning" if prob_sucesso >= 0.35 else "danger"
            emoji = "🏆" if prob_sucesso >= 0.5 else "🎯" if prob_sucesso >= 0.35 else "📉"

            return html.Div([
                html.Div(emoji, style={"fontSize": "2.5rem", "textAlign": "center"}),
                html.H2(
                    f"{prob_sucesso:.1%}",
                    className=f"text-{cor} text-center mb-0 fw-bold",
                    style={"fontSize": "2.8rem"}
                ),
                html.P("de chance de nota ≥ 7.0", className="text-muted text-center mb-2"),
                dbc.Progress(
                    value=prob_sucesso * 100,
                    color=cor,
                    style={"height": "12px"},
                    className="mb-2"
                ),
                html.Small(
                    f"Nota < 7: {prob_falha:.1%}  |  Nota ≥ 7: {prob_sucesso:.1%}",
                    className="text-muted d-block text-center"
                )
            ])

        except Exception as e:
            return dbc.Alert(f"Erro na predição: {str(e)}", color="danger")