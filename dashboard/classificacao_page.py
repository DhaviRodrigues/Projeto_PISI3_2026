import dash
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
LGBM_MODELS_DIR = BASE_DIR / 'classificacao' / 'models'

model_lgbm_caudalonga = None
try:
    caminho_modelo = str(LGBM_MODELS_DIR / 'lgbm_campeao_caudalonga.pkl')
    model_lgbm_caudalonga = joblib.load(caminho_modelo)
    print("✅ Página Analítica: Modelo LightGBM Campeão carregado com sucesso!")
except Exception as e:
    print(f"⚠️ Aviso: Não foi possível carregar o modelo LightGBM. Erro: {e}")

def formatar_milhar(valor):
    if valor is None or valor == "": return ""
    valor_limpo = re.sub(r"[^\d]", "", str(valor))
    if not valor_limpo: return ""
    return f"{int(valor_limpo):,}".replace(",", ".")

def limpar_para_float(valor):
    if valor is None or valor == "": return 0.0
    valor_str = str(valor).replace(".", "").replace(",", ".")
    try:
        return float(valor_str) if valor_str else 0.0
    except:
        return 0.0

def create_lgbm_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Previsão de Sucesso (LightGBM)", className="mb-2 text-dark"),
                html.P("Descubra o potencial do seu projeto na Cauda Longa. Este modelo calcula a probabilidade do seu filme alcançar o status de Sucesso de Crítica (Nota Média / vote_average ≥ 6.5 no IMDb).", className="text-muted fs-5"),
            ], width=12)
        ], className="mb-4"),
        html.Hr(className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Metadados do Filme (Features)", className="mb-0")),
                    dbc.CardBody([
                        html.Label("Orçamento Estimado (USD):", className="fw-bold fs-6"),
                        dbc.Input(id="input-budget", type="text", placeholder="Ex: 1.500.000", className="mb-2"),
                        
                        html.Label("Duração (Minutos):", className="fw-bold fs-6"),
                        dbc.Input(id="input-runtime", type="number", min=1, placeholder="Ex: 110", className="mb-2"),
                        
                        html.Label("Mês de Lançamento Previsto:", className="fw-bold fs-6"),
                        dbc.Select(
                            id="input-month", 
                            options=[
                                {"label": "Janeiro", "value": "1"}, 
                                {"label": "Fevereiro", "value": "2"}, 
                                {"label": "Março", "value": "3"}, 
                                {"label": "Abril", "value": "4"}, 
                                {"label": "Maio", "value": "5"}, 
                                {"label": "Junho", "value": "6"}, 
                                {"label": "Julho", "value": "7"}, 
                                {"label": "Agosto", "value": "8"}, 
                                {"label": "Setembro", "value": "9"}, 
                                {"label": "Outubro", "value": "10"}, 
                                {"label": "Novembro", "value": "11"}, 
                                {"label": "Dezembro", "value": "12"}
                            ], 
                            value="11", 
                            className="mb-3"
                        ),
                        
                        html.Label("Gênero Principal:", className="fw-bold fs-6"),
                        dbc.Select(
                            id="input-genre", 
                            options=[{"label": g, "value": g} for g in ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror', 'Romance', 'Documentary', 'Sci-Fi']], 
                            value="Drama", 
                            className="mb-3"
                        ),
                        
                        html.Label("Idioma Original:", className="fw-bold fs-6"),
                        dbc.Select(
                            id="input-language", 
                            options=[{"label": "Inglês (en)", "value": "en"}, {"label": "Espanhol (es)", "value": "es"}, {"label": "Francês (fr)", "value": "fr"}, {"label": "Outros", "value": "outros"}], 
                            value="en", 
                            className="mb-3"
                        ),
                        
                        html.Label("Possui Tagline de Marketing?", className="fw-bold fs-6"),
                        dbc.Select(
                            id="input-tagline", 
                            options=[{"label": "Sim (1)", "value": "1"}, {"label": "Não (0)", "value": "0"}], 
                            value="1", 
                            className="mb-3"
                        ),

                        html.Label("Tamanho da Sinopse (Caracteres):", className="fw-bold fs-6"),
                        dbc.Input(id="input-overview-len", type="number", min=0, placeholder="Ex: 250", value=250, className="mb-4"),
                                                                                                        
                        dbc.Button("Processar Inferência", id="btn-calcular", color="dark", className="w-100 fw-bold", n_clicks=0),
                        html.Div(id="msg-erro", className="mt-3")
                    ])
                ], className="shadow-sm h-100 border-0 bg-light")
            ], width=3),

            dbc.Col([
                dbc.Card([
                    # ATUALIZADO AQUI PARA 6.5
                    dbc.CardHeader(html.H5("Impacto do Orçamento na Chance de Sucesso (Nota ≥ 6.5)", className="mb-0 text-primary fw-bold")),
                    dbc.CardBody([
                        dcc.Loading(
                            type="dot", color="#212529",
                            children=dcc.Graph(id="grafico-pdp", style={"height": "380px"})
                        )
                    ])
                ], className="shadow-sm h-100 border-0")
            ], width=9)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    # ATUALIZADO AQUI PARA 6.5
                    dbc.CardHeader(html.H5("Quais Gêneros têm mais chance de bater a Nota 6.5?", className="mb-0")),
                    dbc.CardBody([
                        dcc.Loading(
                            type="dot", color="#212529",
                            children=dcc.Graph(id="grafico-sensibilidade", style={"height": "350px"})
                        )
                    ])
                ], className="shadow-sm border-0")
            ], width=7),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Importância Global de Features (Gain)", className="mb-0")),
                    dbc.CardBody([
                        dcc.Graph(id="grafico-importancia", style={"height": "350px"})
                    ])
                ], className="shadow-sm border-0")
            ], width=5)
        ], className="mb-4")
        
    ], style={"marginLeft": "18rem", "marginRight": "2rem", "paddingTop": "1rem", "paddingBottom": "3rem"})

def register_lgbm_callbacks(app):

    @app.callback(
        Output("input-budget", "value"),
        Input("input-budget", "value"),
        prevent_initial_call=True
    )
    def mask_budget(val):
        if not val: return ""
        return formatar_milhar(val)

    @app.callback(
        [
            Output("grafico-pdp", "figure"),
            Output("grafico-sensibilidade", "figure"),
            Output("grafico-importancia", "figure"),
            Output("msg-erro", "children")
        ],
        Input("btn-calcular", "n_clicks"),
        [
            State("input-budget", "value"),
            State("input-runtime", "value"),
            State("input-month", "value"), 
            State("input-genre", "value"),
            State("input-language", "value"),
            State("input-tagline", "value"),
            State("input-overview-len", "value")
        ]
    )
    def atualizar_dashboard_cientifico(n_clicks, budget_raw, runtime, month, genre, language, tagline, overview_len):
        
        TEMPLATE = "simple_white"
        lista_generos_teste = ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror', 'Romance', 'Documentary', 'Sci-Fi']

        fig_imp = go.Figure()
        if model_lgbm_caudalonga is not None and hasattr(model_lgbm_caudalonga, 'feature_importances_'):
            importancias = model_lgbm_caudalonga.feature_importances_
            features = model_lgbm_caudalonga.feature_name_
            df_imp = pd.DataFrame({'Feature': features, 'Importância': importancias})
            df_imp = df_imp.sort_values('Importância', ascending=True).tail(8) 
            
            fig_imp = px.bar(df_imp, x='Importância', y='Feature', orientation='h', color_discrete_sequence=['#495057'])
            fig_imp.update_layout(template=TEMPLATE, margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Peso na Árvore", yaxis_title="")

        if not n_clicks or n_clicks == 0:
            vazio = go.Figure().update_layout(template=TEMPLATE, title="Aguardando metadados de entrada...", xaxis_visible=False, yaxis_visible=False)
            return vazio, vazio, fig_imp, ""

        try:
            budget = limpar_para_float(budget_raw)
            runtime = float(runtime) if runtime else 90.0
            month = float(month) if month else 11.0
            tagline = int(tagline)
            overview_len = float(overview_len) if overview_len else 0.0

            if model_lgbm_caudalonga is None: 
                return dash.no_update, dash.no_update, dash.no_update, dbc.Alert("Modelo ausente. Verifique se o arquivo .pkl está na pasta correta.", color="danger")

            def criar_vetor(b, rt, m, tg, ov_len, g, lang):
                d = {
                    'budget': b,
                    'runtime': rt,
                    'mes_lancamento': m, 
                    'tem_tagline': tg,
                    'tamanho_sinopse': ov_len,
                    f'genero_{g}': 1,
                    f'idioma_{lang}': 1
                }
                df_out = pd.DataFrame([d])
                colunas_do_modelo = model_lgbm_caudalonga.feature_name_
                df_out = df_out.reindex(columns=colunas_do_modelo, fill_value=0)
                return df_out[colunas_do_modelo]

            input_df = criar_vetor(budget, runtime, month, tagline, overview_len, genre, language)
            prob_exata = model_lgbm_caudalonga.predict_proba(input_df)[0][1]
            vetor_budget = np.linspace(0, 150_000_000, 50)
            df_pdp_sim = pd.concat([criar_vetor(b, runtime, month, tagline, overview_len, genre, language) for b in vetor_budget], ignore_index=True)
            probs_pdp = model_lgbm_caudalonga.predict_proba(df_pdp_sim)[:, 1]

            fig_pdp = go.Figure()
            fig_pdp.add_trace(go.Scatter(x=vetor_budget, y=probs_pdp, mode='lines', name='Curva de Dependência', line=dict(color='#2b8cbe', width=3)))
            fig_pdp.add_trace(go.Scatter(x=[budget], y=[prob_exata], mode='markers+text', name='Inferência Atual', 
                                         marker=dict(color='#de2d26', size=12, symbol='diamond'),
                                         text=[f"  {prob_exata*100:.1f}%"], textposition="top center", textfont=dict(color='#de2d26', size=14, family='Arial Black')))
            
            fig_pdp.update_layout(
                template=TEMPLATE, margin=dict(l=0, r=20, t=10, b=0),
                xaxis_title="Orçamento Estimado (USD)", yaxis_title="Probabilidade Prevista (P)",
                yaxis=dict(range=[-0.05, 1.05], tickformat='.0%'),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )

            df_sens_sim = pd.concat([criar_vetor(budget, runtime, month, tagline, overview_len, g_teste, language) for g_teste in lista_generos_teste], ignore_index=True)
            probs_sens = model_lgbm_caudalonga.predict_proba(df_sens_sim)[:, 1]
            
            df_sens = pd.DataFrame({'Classe': lista_generos_teste, 'P(Sucesso)': probs_sens}).sort_values('P(Sucesso)', ascending=True)
            cores = ['#de2d26' if g == genre else '#bdbdbd' for g in df_sens['Classe']]
            fig_sens = px.bar(df_sens, x='P(Sucesso)', y='Classe', orientation='h', color='Classe', color_discrete_sequence=cores)

            fig_sens.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="#de2d26", annotation_text="Limiar de Decisão (50%)")
            
            fig_sens.update_layout(
                template=TEMPLATE, margin=dict(l=0, r=20, t=10, b=0), showlegend=False,
                xaxis_title="Probabilidade (P)", yaxis_title="",
                xaxis=dict(range=[0, 1.05], tickformat='.0%')
            )
            fig_sens.update_traces(marker_line_width=0)

            return fig_pdp, fig_sens, fig_imp, ""

        except Exception as e:
            return dash.no_update, dash.no_update, dash.no_update, dbc.Alert(f"Erro no processamento da IA: {str(e)}", color="danger")