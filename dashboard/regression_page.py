import pandas as pd
import numpy as np
import pickle
import gzip
import re
import ast
from pathlib import Path
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
XGBOOST_MODELS_DIR = BASE_DIR / 'xgboost/models'
RANDOM_FOREST_MODELS_DIR = BASE_DIR / 'random_forest/models'

modelos_preditivos = {
    "xgboost": {"runtime_shorts": None, "runtime_longs": None, "vote": None, "features_rt": [], "features_vote": []},
    "random_forest": {"runtime_shorts": None, "runtime_longs": None, "vote": None, "features_rt": [], "features_vote": []}
}

def formatar_milhar(valor):
    if valor is None or valor == "": return ""
    valor_limpo = re.sub(r"[^\d]", "", str(valor))
    if not valor_limpo: return ""
    return f"{int(valor_limpo):,}".replace(",", ".")

def limpar_para_float(valor):
    if valor is None or valor == "": return None
    valor_str = str(valor).replace(".", "").replace(",", ".")
    try: return float(valor_str) if valor_str else None
    except: return None

def carregar_modelos():
    global modelos_preditivos

    caminhos = {
        ("xgboost", "vote"): XGBOOST_MODELS_DIR / 'xgb_vote_average_model.pkl.gz',
        ("xgboost", "runtime_shorts"): XGBOOST_MODELS_DIR / 'xgb_shorts_runtime_model.pkl.gz',
        ("xgboost", "runtime_longs"): XGBOOST_MODELS_DIR / 'xgb_longs_runtime_model.pkl.gz',
        ("random_forest", "vote"): RANDOM_FOREST_MODELS_DIR / 'random_forest_vote_average_model.pkl.gz',
        ("random_forest", "runtime_shorts"): RANDOM_FOREST_MODELS_DIR / 'random_forest_runtime_shorts_model.pkl.gz',
        ("random_forest", "runtime_longs"): RANDOM_FOREST_MODELS_DIR / 'random_forest_runtime_longs_model.pkl.gz'
    }

    for (alg, tipo), caminho in caminhos.items():
        with gzip.open(str(caminho), 'rb') as f:
            modelos_preditivos[alg][tipo] = pickle.load(f)

    try:
        with open(XGBOOST_MODELS_DIR / 'features_runtime_model.pkl', 'rb') as f:
            modelos_preditivos["xgboost"]["features_rt"] = pickle.load(f)
    except:
        print("Aviso: Features de runtime XGBoost não carregadas")
    
    try:
        with open(XGBOOST_MODELS_DIR / 'features_vote_average_model.pkl', 'rb') as f:
            modelos_preditivos["xgboost"]["features_vote"] = pickle.load(f)
    except:
        print("Aviso: Features de vote XGBoost não carregadas")
    
    try:
        with open(RANDOM_FOREST_MODELS_DIR / 'features_rf_vote_average_model.pkl', 'rb') as f:
            modelos_preditivos["random_forest"]["features_vote"] = pickle.load(f)
    except:
        print("Aviso: Features de vote Random Forest não carregadas")
    
    try:
        with open(RANDOM_FOREST_MODELS_DIR / 'features_runtime_model.pkl', 'rb') as f:
            modelos_preditivos["random_forest"]["features_rt"] = pickle.load(f)
    except:
        print("Aviso: Features de runtime Random Forest não carregadas")

carregar_modelos()

def carregar_paises():
    try:
        df = pd.read_parquet(BASE_DIR / 'clusterizacao' / 'models' / 'Imdb_Movie_Dataset_Clustered.parquet')
        paises_unicos = set()
        
        for countries_str in df['production_countries'].dropna():
            try:
                if isinstance(countries_str, str):
                    if countries_str.startswith('['):
                        countries_list = ast.literal_eval(countries_str)
                    else:
                        countries_list = [c.strip() for c in countries_str.split(',')]
                    
                    if isinstance(countries_list, list):
                        paises_unicos.update(countries_list)
                    else:
                        paises_unicos.add(str(countries_list))
            except:
                pass
        return sorted(list(paises_unicos))
    except Exception as e:
        print(f"Erro ao carregar países: {e}")
        return ["United States of America", "United Kingdom", "France"]

PAISES_OPCOES = [{"label": p, "value": p} for p in carregar_paises()]

def create_regression_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([html.H1("Predições Analíticas de Regressão", className="mb-2 mt-4")], width=6),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Label("Algoritmo:", className="fw-bold mb-0 text-end me-1", style={"width": "150px"}),
                        dbc.RadioItems(id="seletor-algoritmo", className="btn-group", inputClassName="btn-check",
                                       labelClassName="btn btn-outline-secondary", labelCheckedClassName="active",
                                       options=[{"label": "XGBoost", "value": "xgboost"}, {"label": "Random Forest", "value": "random_forest"}],
                                       value="xgboost"),
                    ], className="d-flex align-items-center justify-content-end mb-3"),
                    html.Div([
                        html.Label("Alvo da Predição:", className="fw-bold mb-0 text-end me-1", style={"width": "150px"}),
                        dbc.RadioItems(id="seletor-modelo-reg", className="btn-group", inputClassName="btn-check",
                                       labelClassName="btn btn-outline-primary", labelCheckedClassName="active",
                                       options=[{"label": "Duração do Filme", "value": "runtime"}, {"label": "Avaliação do Filme", "value": "vote_average"}],
                                       value="runtime"),
                    ], className="d-flex align-items-center justify-content-end")
                ], className="w-100")
            ], width=6)
        ], className="mb-4 align-items-end"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Orçamento do Filme (USD):"),
                dbc.Input(id="reg-input-budget", type="text", className="mb-3"),
                html.Label("Contagem Total de Votos:"),
                dbc.Input(id="reg-input-votecount", type="text", className="mb-3"),
                html.Label("Tamanho da Sinopse:"),
                dbc.Input(id="reg-input-overviewlen", type="text", className="mb-3"),
            ], width=4),
            dbc.Col([
                html.Label("Ano de Lançamento:"),
                dbc.Input(id="reg-input-year", type="text", className="mb-3"),
                html.Label("Gênero Principal:"),
                dbc.Select(id="reg-input-genre", options=[{"label": g, "value": g} for g in ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror', 'Romance', 'Adventure', 'Documentary']], value="Action", className="mb-3"),
                html.Label("País de Produção:"),
                dbc.Select(id="reg-input-country", options=PAISES_OPCOES, value=PAISES_OPCOES[0]["value"] if PAISES_OPCOES else "United States of America", className="mb-3"),
            ], width=4),
            dbc.Col([
                html.Label("Idioma é Inglês?"),
                dbc.Select(id="reg-input-isen", options=[{"label": "Sim", "value": "1"}, {"label": "Não", "value": "0"}], value="1", className="mb-3"),
                html.Div(id="bloco-input-nota-media", children=[
                    html.Label("Nota Avaliação:"),
                    dbc.Input(id="reg-input-voteaverage-dinamico", type="text", className="mb-3"),
                ], style={"display": "block"}),
                html.Div(id="bloco-input-duracao", children=[
                    html.Label("Duração do Filme:"),
                    dbc.Input(id="reg-input-runtime-dinamico", type="text", className="mb-3"),
                ], style={"display": "none"}),
                html.Div(id="bloco-input-short", children=[
                    html.Label("É um Filme Curta-Metragem?"),
                    dbc.Select(id="reg-input-isshort", options=[{"label": "Sim", "value": "1"}, {"label": "Não", "value": "0"}], value="0", className="mb-3"),
                ], style={"display": "none"}),
            ], width=4),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5(id="reg-titulo-card-resultado", children="Previsão de Duração")),
                    dbc.CardBody([
                        dbc.Button("Calcular Predição", id="btn-calcular-reg", color="primary", className="mb-3 w-100", n_clicks=0),
                        html.Div(id="msg-erro-reg"),
                        html.Div(id="txt-resultado-reg", className="text-center fw-bold fs-5")
                    ])
                ], className="shadow-sm mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Explicabilidade Visual com SHAP"),
                dbc.Tabs([
                    dbc.Tab(label="Tabela", tab_id="tab-tabela", children=[html.Div([dash_table.DataTable(id="tabela-importancia-features", page_size=10)], className="p-3")]),
                    dbc.Tab(label="Bar Plot", tab_id="tab-bar", children=[dcc.Graph(id="grafico-shap-bar-interativo")]),
                    dbc.Tab(label="Summary Plot", tab_id="tab-summary", children=[dcc.Graph(id="grafico-shap-summary-interativo")]),
                ], id="abas-shap", active_tab="tab-tabela")
            ])
        ])
    ], style={"marginLeft": "18rem", "marginRight": "2rem", "paddingTop": "1rem"})

def register_regression_callbacks(app):
    @app.callback(
        [Output("bloco-input-nota-media", "style"), Output("bloco-input-duracao", "style"), Output("bloco-input-short", "style"), Output("reg-titulo-card-resultado", "children")],
        Input("seletor-modelo-reg", "value")
    )
    def alternar_interface(modelo_selecionado):
        if modelo_selecionado == "runtime":
            return {"display": "block"}, {"display": "none"}, {"display": "block"}, "Previsão de Duração"
        return {"display": "none"}, {"display": "block"}, {"display": "none"}, "Previsão de Nota Média"

    @app.callback(
        [Output("tabela-importancia-features", "data"), 
         Output("grafico-shap-bar-interativo", "figure"), 
         Output("grafico-shap-summary-interativo", "figure")],
        [Input("seletor-algoritmo", "value"), Input("seletor-modelo-reg", "value")]
    )
    def atualizar_analise_explicativa_shap(algoritmo, modelo):
        prefixo = "xgb" if algoritmo == "xgboost" else "rf"
        alvo = "runtime" if modelo == "runtime" else "vote"
        pasta_base = XGBOOST_MODELS_DIR if algoritmo == "xgboost" else RANDOM_FOREST_MODELS_DIR
        caminho_arquivo = pasta_base / "shap" / f"{prefixo}_{alvo}_shap_data.pkl"
        
        try:
            with open(caminho_arquivo, 'rb') as f:
                data = pickle.load(f)
            shap_matrix = np.array(data['shap_values'])
            mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
            
            df_shap = pd.DataFrame({'Feature': data['features'], 'Importance': mean_abs_shap})
            records = df_shap.sort_values(by='Importance', ascending=False).to_dict('records')
            
            df_top = df_shap.sort_values(by='Importance').tail(12)
            fig_bar = go.Figure(go.Bar(x=df_top['Importance'], y=df_top['Feature'], orientation='h'))
            fig_bar.update_layout(template="plotly_white", margin=dict(l=150, r=20, t=10, b=40))
            
            fig_sum = go.Figure()
            n = min(200, len(shap_matrix))
            idx = np.random.choice(len(shap_matrix), n, replace=False)
            
            limit = np.percentile(np.abs(shap_matrix), 99)
            if limit == 0: limit = 1.0

            if shap_matrix.ndim == 1:
                shap_matrix = shap_matrix.reshape(1, -1)
            
            top_features_indices = df_top.index.tolist()
            for idx_feat in top_features_indices:
                feat = data['features'][idx_feat]
                fig_sum.add_trace(go.Scatter(
                    x=shap_matrix[idx, idx_feat], 
                    y=[feat]*n, 
                    mode='markers', 
                    marker=dict(
                        color=shap_matrix[idx, idx_feat],
                        colorscale='RdBu',
                        cmin=-limit,
                        cmax=limit,
                        size=10,
                        opacity=1.0,
                        line=dict(width=0.8, color='black')
                    ),
                    name=feat,
                    hovertemplate=f"<b>{feat}</b><br>Impacto SHAP: %{{x:.3f}}<extra></extra>"
                ))
            
            fig_sum.update_layout(template="plotly_white", height=600, showlegend=False, xaxis_title="Impacto SHAP (Vermelho = Negativo, Azul = Positivo)", margin=dict(l=150, r=20, t=30, b=50))
            
            return records, fig_bar, fig_sum
            
        except Exception as e:
            print(f"ERRO CRÍTICO NO SHAP: {e}")
            return [], go.Figure(), go.Figure()
        
    @app.callback(
        [Output("txt-resultado-reg", "children", allow_duplicate=True), 
         Output("msg-erro-reg", "children"), 
         Output("btn-calcular-reg", "disabled")],
        Input("btn-calcular-reg", "n_clicks"),
        [
            State("seletor-algoritmo", "value"), 
            State("seletor-modelo-reg", "value"),
            State("reg-input-budget", "value"), 
            State("reg-input-votecount", "value"),
            State("reg-input-year", "value"), 
            State("reg-input-genre", "value"),
            State("reg-input-country", "value"),
            State("reg-input-isen", "value"),
            State("reg-input-voteaverage-dinamico", "value"), 
            State("reg-input-runtime-dinamico", "value"),
            State("reg-input-isshort", "value"), 
            State("reg-input-overviewlen", "value")
        ],
        prevent_initial_call=True
    )
    def processar_predicao(n_clicks, algoritmo, modelo, budget_raw, votecount_raw, year_raw, genre, country, is_en, vote_avg_din_raw, runtime_din_raw, is_short, overview_len_raw):
        
        if not n_clicks or n_clicks == 0: return "---", "", False
        
        if (modelo == "runtime" and not modelos_preditivos[algoritmo]["runtime_longs"]) or \
        (modelo == "vote_average" and not modelos_preditivos[algoritmo]["vote"]):
            return "---", dbc.Alert("Erro: Modelo não encontrado na memória.", color="danger"), False
        
        features_key = "features_rt" if modelo == "runtime" else "features_vote"
        if not modelos_preditivos[algoritmo][features_key]:
            return "---", dbc.Alert(f"Erro: Features não carregadas.", color="danger"), False
        
        try:
            erros_validacao = []
            budget = limpar_para_float(budget_raw) if budget_raw else 0.0
            vote_count = limpar_para_float(votecount_raw)
            
            try: year = int(re.sub(r"\D", "", str(year_raw))) if year_raw else None
            except: year = None

            if vote_count is None: erros_validacao.append("Contagem de Votos")
            if year is None: erros_validacao.append("Ano de Lançamento")

            vote_avg_din, overview_len, runtime_din = None, None, None

            if modelo == "runtime":
                avg_clean = str(vote_avg_din_raw).replace(",", ".") if vote_avg_din_raw else ""
                try: vote_avg_din = float(re.sub(r"[^\d.]", "", avg_clean)) if avg_clean else None
                except: vote_avg_din = None
                if vote_avg_din is None: erros_validacao.append("Nota Média do Filme")
            else:
                try:
                    overview_len = int(re.sub(r"\D", "", str(overview_len_raw))) if overview_len_raw else None
                    runtime_din = float(re.sub(r"[^\d.]", "", str(runtime_din_raw).replace(",", "."))) if runtime_din_raw else None
                except: pass
                if overview_len is None: erros_validacao.append("Tamanho da Sinopse")
                if runtime_din is None: erros_validacao.append("Duração do Filme")

            if erros_validacao:
                return "---", dbc.Alert([html.Strong("Campos obrigatórios ausentes:"), html.Br(), f"{', '.join(erros_validacao)}"], color="danger", className="mb-2"), False

            if modelo == "vote_average" and int(vote_count) < 5:
                return "---", dbc.Alert("Mínimo de 5 votos necessários.", color="warning", className="mb-2"), False

            budget = min(budget, 500000000.0)
            vote_count = min(vote_count, 50000.0)
            year = max(1880, min(year, 2026))
            periodo_5_anos = (year // 5) * 5
            has_b = 1 if budget > 0 else 0
            is_short_val = int(is_short) if modelo == "runtime" else (1 if (runtime_din and runtime_din < 40) else 0)
            overview_len_val = int(overview_len) if (modelo == "vote_average" and overview_len is not None) else 250
            movie_age_val = 2027 - year
            votes_per_year_val = vote_count / (movie_age_val + 1)

            cpi_history = {
                1913: 9.9,   1914: 10.0,  1915: 10.1,  1916: 10.9,  1917: 12.8,  1918: 15.1,  1919: 17.3,
                1920: 20.0,  1921: 17.9,  1922: 16.8,  1923: 17.1,  1924: 17.1,  1925: 17.5,  1926: 17.7,
                1927: 17.4,  1928: 17.1,  1929: 17.1,  1930: 16.7,  1931: 15.2,  1932: 13.7,  1933: 13.0,
                1934: 13.4,  1935: 13.7,  1936: 13.9,  1937: 14.4,  1938: 14.1,  1939: 13.9,  1940: 14.0,
                1941: 14.7,  1942: 16.3,  1943: 17.3,  1944: 17.6,  1945: 18.0,  1946: 19.5,  1947: 22.3,
                1948: 24.1,  1949: 23.8,  1950: 24.1,  1951: 26.0,  1952: 26.5,  1953: 26.7,  1954: 26.9,
                1955: 26.8,  1956: 27.2,  1957: 28.1,  1958: 28.9,  1959: 29.1,  1960: 29.6,  1961: 29.9,
                1962: 30.2,  1963: 30.6,  1964: 31.0,  1965: 31.5,  1966: 32.4,  1967: 33.4,  1968: 34.8,
                1969: 36.7,  1970: 38.8,  1971: 40.5,  1972: 41.8,  1973: 44.4,  1974: 49.3,  1975: 53.8,
                1976: 56.9,  1977: 60.6,  1978: 65.2,  1979: 72.6,  1980: 82.4,  1981: 90.9,  1982: 96.5,
                1983: 99.6,  1984: 103.9, 1985: 107.6, 1986: 109.6, 1987: 113.6, 1988: 118.3, 1989: 124.0,
                1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2, 1995: 152.4, 1996: 156.9,
                1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0,
                2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.34, 2008: 215.30, 2009: 214.54, 2010: 218.06,
                2011: 224.94, 2012: 229.59, 2013: 232.96, 2014: 236.74, 2015: 237.02, 2016: 240.01, 2017: 245.12,
                2018: 251.11, 2019: 255.66, 2020: 258.81, 2021: 270.97, 2022: 292.66, 2023: 304.70, 2024: 313.20,
                2025: 320.10, 2026: 326.50
            }
            mult = cpi_history.get(year, cpi_history.get((year // 10) * 10, 1.0))
            budget_inflacionado = budget * (cpi_history[2026] / mult)
            log_budget_val = np.log1p(budget_inflacionado)

            if modelo == "runtime":
                v_avg = float(vote_avg_din)
                data_dict = {'vote_average': v_avg, 'vote_count': vote_count, 'log_vote_count': np.log1p(vote_count), 
                            'release_year': year, 'release_5_years': periodo_5_anos, 'movie_age': movie_age_val, 
                            'budget': budget, 'is_short_keyword': is_short_val, 'overview_len': overview_len_val, 
                            'tagline_len': 45, 'genre_mean': 100.0, 'genre_decade_mean': 100.0, 
                            'vote_score_interact': v_avg * np.log1p(vote_count), 'has_budget': has_b, 
                            'is_en': int(is_en), 'votes_per_year': votes_per_year_val}
                
                features = modelos_preditivos[algoritmo]["features_rt"]
                for col in features:
                    if col not in data_dict:
                        if col.startswith("genres_"):
                            feat_val = col.replace("genres_", "")
                            data_dict[col] = 1 if genre.lower() == feat_val.lower() else 0
                        elif col.startswith("production_countries_"):
                            feat_val = col.replace("production_countries_", "")
                            data_dict[col] = 1 if country.lower() == feat_val.lower() else 0
                        else:
                            data_dict[col] = 0
                            
                input_df = pd.DataFrame([data_dict]).reindex(columns=features, fill_value=0)
                pred_val = modelos_preditivos[algoritmo]["runtime_shorts"].predict(input_df)[0] if is_short_val == 1 else modelos_preditivos[algoritmo]["runtime_longs"].predict(input_df)[0]
                
                if algoritmo == "random_forest":
                    pred_val = np.expm1(pred_val)
                    
                return html.Span(f"{pred_val:.1f} minutos", className="text-primary fw-bold fs-5"), "", False
                
            else:
                rt_val = float(runtime_din)
                data_dict = {'runtime': rt_val, 'vote_count': vote_count, 'log_vote_count': np.log1p(vote_count),
                            'release_year': year, 'release_5_years': periodo_5_anos, 'movie_age': movie_age_val,
                            'budget': budget, 'log_budget': np.log1p(budget), 'is_short_keyword': is_short_val, 
                            'overview_len': overview_len_val, 'tagline_len': 45, 'has_overview': 1, 'has_tagline': 1,
                            'genre_vote_mean': 6.2, 'genre_decade_vote_mean': 6.2, 'has_budget': has_b, 
                            'is_en': int(is_en), 'votes_per_year': votes_per_year_val}

                features = modelos_preditivos[algoritmo]["features_vote"]
                for col in features:
                    if col not in data_dict:
                        if col.startswith("genres_"):
                            feat_val = col.replace("genres_", "")
                            data_dict[col] = 1 if genre.lower() == feat_val.lower() else 0
                        elif col.startswith("production_countries_"):
                            feat_val = col.replace("production_countries_", "")
                            data_dict[col] = 1 if country.lower() == feat_val.lower() else 0
                        else:
                            data_dict[col] = 0

                input_df = pd.DataFrame([data_dict]).reindex(columns=features, fill_value=0)
                pred_val = modelos_preditivos[algoritmo]["vote"].predict(input_df)[0]
                return html.Span(f"{pred_val:.2f} / 10", className="text-success fw-bold fs-5"), "", False

        except Exception as e:
            return "---", dbc.Alert(f"Erro no processamento: {str(e)}", color="danger", className="mb-2"), False