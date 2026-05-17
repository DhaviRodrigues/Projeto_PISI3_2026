import pandas as pd
import numpy as np
import pickle
import gzip
import re
from pathlib import Path
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'xgboost' / 'models'

model_rt_shorts = None
model_rt_longs = None
features_rt = []
model_vote = None
features_vote = []

with gzip.open(str(MODELS_DIR / 'xgb_shorts_runtime_model.pkl.gz'), 'rb') as f:
    model_rt_shorts = pickle.load(f)
with gzip.open(str(MODELS_DIR / 'xgb_longs_runtime_model.pkl.gz'), 'rb') as f:
    model_rt_longs = pickle.load(f)
with open(str(MODELS_DIR / 'features_runtime_model.pkl'), 'rb') as f:
    features_rt = pickle.load(f)

with gzip.open(str(MODELS_DIR / 'xgb_vote_average_model.pkl.gz'), 'rb') as f:
    model_vote = pickle.load(f)
with open(str(MODELS_DIR / 'features_vote_average_model.pkl'), 'rb') as f:
    features_vote = pickle.load(f)

def formatar_milhar(valor):
    """Formata número para padrão brasileiro xx.xxx.xxx"""
    if valor is None or valor == "":
        return ""
    valor_limpo = re.sub(r"[^\d]", "", str(valor))
    if not valor_limpo:
        return ""
    num = int(valor_limpo)
    return f"{num:,}".replace(",", ".")

def limpar_para_float(valor):
    """Converte valor formatado para float"""
    if valor is None or valor == "":
        return None
    valor_str = str(valor).replace(".", "")
    valor_str = valor_str.replace(",", ".")
    try:
        return float(valor_str) if valor_str else None
    except:
        return None

def create_xgboost_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Predições Analíticas com XGBoost", className="mb-2"),
                html.P("Selecione o objetivo da modelagem para ajustar a interface de simulação.", className="text-muted"),
            ], width=7),
            dbc.Col([
                html.Div([
                    html.Label("Alvo da Predição:", className="fw-bold mb-2 d-block text-end"),
                    dbc.RadioItems(
                        id="seletor-modelo-xgb",
                        className="btn-group float-end",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=[
                            {"label": "Duração do Filme", "value": "runtime"},
                            {"label": "Média de Votos", "value": "vote_average"},
                        ],
                        value="runtime",
                    )
                ], className="d-inline-block w-100")
            ], width=5, className="d-flex align-items-center justify-content-end")
        ], className="mb-4 align-items-end"),
        html.Hr(className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Orçamento do Filme (Budget em USD) * :"),
                dbc.Input(id="input-budget", type="text", placeholder="Ex: 50.000.000", className="mb-3"),
                html.Label("Receita do Filme (Revenue em USD) * :"),
                dbc.Input(id="input-revenue", type="text", placeholder="Ex: 120.000.000", className="mb-3"),
                html.Label("Popularidade TMDB (Popularity) * :"),
                dbc.Input(id="input-popularity", type="text", placeholder="Ex: 1.6 ou 25.5", className="mb-3"),
            ], width=4),
            
            dbc.Col([
                html.Label("Contagem Total de Votos (Vote Count) * :"),
                dbc.Input(id="input-votecount", type="text", placeholder="Ex: 1.500", className="mb-3"),
                html.Label("Ano de Lançamento (Release Year) * :"),
                dbc.Input(id="input-year", type="text", placeholder="Ex: 2024", className="mb-3"),
                html.Label("Gênero Principal do Filme * :"),
                dbc.Select(id="input-genre", options=[{"label": g, "value": g} for g in ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror', 'Romance', 'Adventure', 'Documentary']], value="Action", className="mb-3"),
            ], width=4),
            
            dbc.Col([
                html.Label("Idioma Original é Inglês (en)?"),
                dbc.Select(id="input-isen", options=[{"label": "Sim", "value": "1"}, {"label": "Não", "value": "0"}], value="1", className="mb-3"),
                
                html.Div(id="bloco-especifico-runtime", children=[
                    html.Label("Contem a palavra-chave 'Short'?"),
                    dbc.Select(id="input-isshort", options=[{"label": "Sim", "value": "1"}, {"label": "Não", "value": "0"}], value="0", className="mb-3"),
                    html.Label("Nota Média do Filme (Vote Average 0-10) * :"),
                    dbc.Input(id="input-voteaverage-dinamico", type="text", placeholder="Ex: 7.2", className="mb-3"),
                ]),
                
                html.Div(id="bloco-especifico-vote", children=[
                    html.Label("Tamanho da Sinopse (Overview Characters) * :"),
                    dbc.Input(id="input-overviewlen", type="text", placeholder="Ex: 250", className="mb-3"),
                    html.Label("Duração Real do Filme (Runtime em minutos) * :"),
                    dbc.Input(id="input-runtime-dinamico", type="text", placeholder="Ex: 110", className="mb-3"),
                ], style={"display": "none"})
            ], width=4)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5(id="titulo-card-resultado", children="Previsão de Duração")),
                    dbc.CardBody([
                        dbc.Button(
                            "Calcular Predição", 
                            id="btn-calcular-xgb", 
                            color="primary", 
                            className="mb-3 w-100", 
                            n_clicks=0,
                            disabled=False
                        ),
                        html.Div(id="msg-erro-xgb", className="mb-2"),
                        dcc.Loading(
                            id="loading-resultado-xgb",
                            type="default",
                            color="#0d6efd",
                            children=html.Div(id="txt-resultado-xgb", className="text-center fw-bold fs-5", children="...")
                        )
                    ])
                ], id="card-resultado-xgb", outline=True, color="primary", className="shadow-sm")
            ], width=12)
        ], className="mt-4")
    ], style={"marginLeft": "18rem", "marginRight": "2rem", "paddingTop": "1rem"})

def register_xgboost_callbacks(app):
    """Registra todos os callbacks da página XGBoost"""
    pass

    @app.callback(
        Output("input-budget", "value"),
        Input("input-budget", "value"),
        prevent_initial_call=True
    )
    def mask_budget(val):
        if not val: return ""
        return formatar_milhar(val)

    @app.callback(
        Output("input-revenue", "value"),
        Input("input-revenue", "value"),
        prevent_initial_call=True
    )
    def mask_revenue(val):
        if not val: return ""
        return formatar_milhar(val)

    @app.callback(
        Output("input-votecount", "value"),
        Input("input-votecount", "value"),
        prevent_initial_call=True
    )
    def mask_votecount(val):
        if not val: return ""
        return formatar_milhar(val)
    
    @app.callback(
        [
            Output("bloco-especifico-runtime", "style"),
            Output("bloco-especifico-vote", "style"),
            Output("titulo-card-resultado", "children"),
            Output("card-resultado-xgb", "color"),
            Output("btn-calcular-xgb", "color"),
            Output("txt-resultado-xgb", "children")
        ],
        Input("seletor-modelo-xgb", "value")
    )
    def alternar_interface_por_modelo(modelo_selecionado):
        if modelo_selecionado == "runtime":
            return (
                {"display": "block"}, 
                {"display": "none"}, 
                "Previsão de Duração", 
                "primary", 
                "primary",
                "..."
            )
        else:
            return (
                {"display": "none"}, 
                {"display": "block"}, 
                "Previsão de Nota Média", 
                "primary", 
                "primary",
                "..."
            )

    @app.callback(
        [
            Output("txt-resultado-xgb", "children", allow_duplicate=True),
            Output("msg-erro-xgb", "children"),
            Output("btn-calcular-xgb", "disabled")
        ],
        Input("btn-calcular-xgb", "n_clicks"),
        [
            State("seletor-modelo-xgb", "value"),
            State("input-budget", "value"), State("input-revenue", "value"),
            State("input-popularity", "value"), State("input-votecount", "value"),
            State("input-year", "value"), State("input-genre", "value"),
            State("input-isen", "value"),
            State("input-voteaverage-dinamico", "value"),
            State("input-runtime-dinamico", "value"),
            State("input-isshort", "value"),
            State("input-overviewlen", "value")
        ],
        prevent_initial_call=True
    )
    def processar_predicao_global(n_clicks, modelo, budget_raw, revenue_raw, popularity_raw, votecount_raw, year_raw, genre, is_en, vote_avg_din_raw, runtime_din_raw, is_short, overview_len_raw):
        if not n_clicks or n_clicks == 0:
            return "---", "", False
            
        try:
            erros_validacao = []
            
            budget = limpar_para_float(budget_raw)
            revenue = limpar_para_float(revenue_raw)
            vote_count = limpar_para_float(votecount_raw)
            
            pop_clean = str(popularity_raw).replace(",", ".") if popularity_raw else ""
            try:
                popularity = float(re.sub(r"[^\d.]", "", pop_clean)) if pop_clean else None
            except:
                popularity = None

            try:
                year = int(re.sub(r"\D", "", str(year_raw))) if year_raw else None
            except:
                year = None

            if budget is None: erros_validacao.append("Orçamento")
            if revenue is None: erros_validacao.append("Receita")
            if popularity is None: erros_validacao.append("Popularidade")
            if vote_count is None: erros_validacao.append("Contagem de Votos")
            if year is None: erros_validacao.append("Ano de Lançamento")

            vote_avg_din = None
            overview_len = None
            runtime_din = None

            if modelo == "runtime":
                avg_clean = str(vote_avg_din_raw).replace(",", ".") if vote_avg_din_raw else ""
                try:
                    vote_avg_din = float(re.sub(r"[^\d.]", "", avg_clean)) if avg_clean else None
                except:
                    vote_avg_din = None
                if vote_avg_din is None: erros_validacao.append("Nota Média do Filme")
            else:
                try:
                    overview_len = int(re.sub(r"\D", "", str(overview_len_raw))) if overview_len_raw else None
                    runtime_din = float(re.sub(r"[^\d.]", "", str(runtime_din_raw).replace(",", "."))) if runtime_din_raw else None
                except:
                    pass
                if overview_len is None: erros_validacao.append("Tamanho da Sinopse")
                if runtime_din is None: erros_validacao.append("Duração Real do Filme")

            if erros_validacao:
                msg_erro = dbc.Alert(
                    [
                        html.Strong("Campos obrigatórios ausentes:"),
                        html.Br(),
                        f"{', '.join(erros_validacao)}"
                    ],
                    color="danger",
                    className="mb-2"
                )
                return "---", msg_erro, False

            if modelo == "runtime" and (model_rt_shorts is None or model_rt_longs is None):
                msg_erro = dbc.Alert("Erro: Modelos de duração não encontrados.", color="danger", className="mb-2")
                return "---", msg_erro, False
            if modelo == "vote_average" and model_vote is None:
                msg_erro = dbc.Alert("Erro: Modelo de nota não encontrado.", color="danger", className="mb-2")
                return "---", msg_erro, False

            budget = min(budget, 500000000.0)
            revenue = min(revenue, 3000000000.0)
            popularity = min(popularity, 5000.0)
            vote_count = min(vote_count, 50000.0)
            
            if year > 2026: year = 2026
            elif year < 1880: year = 1880

            decade = (year // 10) * 10
            has_b = 1 if budget > 0 else 0
            has_r = 1 if revenue > 0 else 0
            is_short_val = int(is_short) if (modelo == "runtime" and is_short is not None) else (1 if (runtime_din and runtime_din < 40) else 0)
            overview_len_val = int(overview_len) if (modelo == "vote_average" and overview_len is not None) else 250
            tagline_len_val = 45
            anos_decorridos = 2027 - year + 1

            if modelo == "runtime":
                v_avg = float(vote_avg_din)
                g_mean, gd_mean = 100.0, 100.0
                br_ratio = budget / (revenue + 1.0)
                b_pop = budget / (popularity + 1.0)
                pop_votes = popularity / (vote_count + 1.0)
                vote_interact = v_avg * np.log1p(vote_count)
                pop_genre = popularity * g_mean
                v_per_year = vote_count / anos_decorridos
                
                data_dict = {
                    'vote_average': v_avg, 'vote_count': vote_count, 'release_year': year,
                    'release_decade': decade, 'budget': budget, 'popularity': popularity,
                    'is_short_keyword': is_short_val, 'overview_len': overview_len_val, 'tagline_len': tagline_len_val,
                    'genre_mean': g_mean, 'genre_decade_mean': gd_mean, 'budget_revenue_ratio': br_ratio,
                    'budget_to_popularity': b_pop, 'popularity_to_votes': pop_votes, 'vote_score_interact': vote_interact,
                    'has_budget': has_b, 'has_revenue': has_r, 'is_en': int(is_en), 'pop_genre_interact': pop_genre,
                    'votes_per_year': v_per_year
                }
                
                for col in features_rt:
                    data_dict[col] = 1 if ("genres_" in col and genre.lower() in col.lower()) else data_dict.get(col, 0)
                            
                input_df = pd.DataFrame([data_dict]).reindex(columns=features_rt, fill_value=0)
                pred_log = model_rt_shorts.predict(input_df)[0] if is_short_val == 1 else model_rt_longs.predict(input_df)[0]
                resultado = html.Span(f"{np.expm1(pred_log):.1f} minutos", className="text-primary fs-5")
                return resultado, "", False
                
            else:
                if int(vote_count) < 10:
                    msg_erro = dbc.Alert("  Mínimo de 10 votos necessários para calcular.", color="warning", className="mb-2")
                    return "---", msg_erro, False
                    
                rt_val = float(runtime_din)
                g_v_mean, gd_v_mean = 6.2, 6.2
                log_v = np.log1p(vote_count)
                br_ratio = budget / (revenue + 1.0)
                b_pop = budget / (popularity + 1.0)
                pop_votes = popularity / (vote_count + 1.0)
                v_per_year = vote_count / anos_decorridos
                pop_log_v = popularity * log_v
                
                data_dict = {
                    'runtime': rt_val, 'vote_count': vote_count, 'log_vote_count': log_v,
                    'release_year': year, 'release_decade': decade, 'budget': budget,
                    'popularity': popularity, 'is_short_keyword': is_short_val, 'overview_len': overview_len_val,
                    'tagline_len': tagline_len_val, 'genre_vote_mean': g_v_mean, 'genre_decade_vote_mean': gd_v_mean,
                    'budget_revenue_ratio': br_ratio, 'budget_to_popularity': b_pop, 'popularity_to_votes': pop_votes,
                    'has_budget': has_b, 'has_revenue': has_r, 'is_en': int(is_en), 'votes_per_year': v_per_year,
                    'popularity_log_votes_interact': pop_log_v
                }
                
                for col in features_vote:
                    data_dict[col] = 1 if ("genres_" in col and genre.lower() in col.lower()) else data_dict.get(col, 0)
                            
                input_df = pd.DataFrame([data_dict]).reindex(columns=features_vote, fill_value=0)
                resultado = html.Span(f"{model_vote.predict(input_df)[0]:.2f} / 10", className="text-success fs-5")
                return resultado, "", False
        except Exception as e:
            msg_erro = dbc.Alert(f"Erro no processamento: {str(e)}", color="danger", className="mb-2")
            return "---", msg_erro, False