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

model_lgbm_nicho = None
try:
    model_lgbm_nicho = joblib.load(str(LGBM_MODELS_DIR / 'lgbm_sucesso_nicho_model.pkl'))
    print("Página Analítica: Modelo LightGBM carregado com sucesso!")
except Exception as e:
    print(f"Aviso: Não foi possível carregar o modelo LightGBM: {e}")

def formatar_milhar(valor):
    if valor is None or valor == "": return ""
    valor_limpo = re.sub(r"[^\d]", "", str(valor))
    if not valor_limpo: return ""
    return f"{int(valor_limpo):,}".replace(",", ".")

def limpar_para_float(valor):
    if valor is None or valor == "": return None
    valor_str = str(valor).replace(".", "").replace(",", ".")
    try:
        return float(valor_str) if valor_str else None
    except:
        return None

def create_lgbm_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Diagnóstico do Classificador LightGBM", className="mb-2 text-dark"),
                html.P("Ambiente de explicabilidade do modelo preditivo para a classe 'Sucesso de Nicho'. Utilize o painel para analisar fronteiras de decisão e dependência parcial das variáveis.", className="text-muted"),
            ], width=12)
        ], className="mb-4"),
        html.Hr(className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Vetor de Entrada (Features)", className="mb-0")),
                    dbc.CardBody([
                        html.Label("Receita Contínua (USD):", className="fw-bold"),
                        dbc.Input(id="input-revenue-nicho", type="text", placeholder="Ex: 5.000.000", className="mb-3"),
                        
                        html.Label("Classe Categórica (Gênero):", className="fw-bold"),
                        dbc.Select(
                            id="input-genre-nicho", 
                            options=[{"label": g, "value": g} for g in ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror', 'Romance', 'Adventure', 'Documentary']], 
                            value="Drama", 
                            className="mb-3"
                        ),
                        
                        html.Label("Idioma Original do Filme:", className="fw-bold"),
                        dbc.Select(
                            id="input-isen-nicho", 
                            options=[
                                {"label": "Inglês", "value": "1"}, 
                                {"label": "Outros Idiomas", "value": "0"}
                            ], 
                            value="1", 
                            className="mb-4"
                        ),
                                                
                        dbc.Button("Processar Inferência", id="btn-calcular-nicho", color="dark", className="w-100", n_clicks=0),
                        html.Div(id="msg-erro-nicho", className="mt-3")
                    ])
                ], className="shadow-sm h-100 border-0 bg-light")
            ], width=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Curva de Dependência Parcial (PDP) - Receita vs. Probabilidade", className="mb-0")),
                    dbc.CardBody([
                        dcc.Loading(
                            type="dot",
                            color="#212529",
                            children=dcc.Graph(id="grafico-pdp", style={"height": "380px"})
                        )
                    ])
                ], className="shadow-sm h-100 border-0")
            ], width=9)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Análise de Sensibilidade Categórica (Gêneros)", className="mb-0")),
                    dbc.CardBody([
                        dcc.Loading(
                            type="dot",
                            color="#212529",
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
        Output("input-revenue-nicho", "value"),
        Input("input-revenue-nicho", "value"),
        prevent_initial_call=True
    )
    def mask_revenue(val):
        if not val: return ""
        return formatar_milhar(val)

    @app.callback(
        [
            Output("grafico-pdp", "figure"),
            Output("grafico-sensibilidade", "figure"),
            Output("grafico-importancia", "figure"),
            Output("msg-erro-nicho", "children")
        ],
        Input("btn-calcular-nicho", "n_clicks"),
        [
            State("input-revenue-nicho", "value"),
            State("input-genre-nicho", "value"),
            State("input-isen-nicho", "value")
        ]
    )
    def atualizar_dashboard_cientifico(n_clicks, revenue_raw, genre, is_en):
        
        TEMPLATE = "simple_white"
        lista_generos = ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror', 'Romance', 'Adventure', 'Documentary']

        fig_imp = go.Figure()
        if model_lgbm_nicho is not None and hasattr(model_lgbm_nicho, 'feature_importances_'):
            importancias = model_lgbm_nicho.feature_importances_
            features = model_lgbm_nicho.feature_name_
            df_imp = pd.DataFrame({'Feature': features, 'Importância': importancias})
            df_imp = df_imp.sort_values('Importância', ascending=True).tail(8)
            
            fig_imp = px.bar(df_imp, x='Importância', y='Feature', orientation='h', color_discrete_sequence=['#495057'])
            fig_imp.update_layout(template=TEMPLATE, margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Peso (Split/Gain)", yaxis_title="")

        if not n_clicks or n_clicks == 0:
            vazio = go.Figure().update_layout(template=TEMPLATE, title="Aguardando vetor de entrada...", xaxis_visible=False, yaxis_visible=False)
            return vazio, vazio, fig_imp, ""
            
        try:
            revenue = limpar_para_float(revenue_raw)
            if revenue is None: return dash.no_update, dash.no_update, dash.no_update, dbc.Alert("Receita nula.", color="danger")
            if model_lgbm_nicho is None: return dash.no_update, dash.no_update, dash.no_update, dbc.Alert("Modelo ausente.", color="danger")

            def criar_vetor(r, g, idioma):
                d = {'revenue': r, 'idioma_en': 1 if idioma == "1" else 0, 'idioma_outros': 1 if idioma == "0" else 0}
                for gen in lista_generos:
                    d[f'genero_{gen}'] = 1 if gen == g else 0
                df_out = pd.DataFrame([d])
                if hasattr(model_lgbm_nicho, 'feature_name_'):
                    df_out = df_out.reindex(columns=model_lgbm_nicho.feature_name_, fill_value=0)
                return df_out

            input_df = criar_vetor(revenue, genre, is_en)
            prob_exata = model_lgbm_nicho.predict_proba(input_df)[0][1]

            vetor_receitas = np.linspace(100_000, 150_000_000, 100)
            df_pdp_sim = pd.concat([criar_vetor(r, genre, is_en) for r in vetor_receitas], ignore_index=True)
            probs_pdp = model_lgbm_nicho.predict_proba(df_pdp_sim)[:, 1]

            fig_pdp = go.Figure()
            fig_pdp.add_trace(go.Scatter(x=vetor_receitas, y=probs_pdp, mode='lines', name='Curva de Dependência', line=dict(color='#2b8cbe', width=3)))
            fig_pdp.add_trace(go.Scatter(x=[revenue], y=[prob_exata], mode='markers+text', name='Inferência Atual', 
                                         marker=dict(color='#de2d26', size=12, symbol='diamond'),
                                         text=[f"  {prob_exata*100:.1f}%"], textposition="middle right", textfont=dict(color='#de2d26', size=14, family='Arial Black')))
            
            fig_pdp.update_layout(
                template=TEMPLATE, margin=dict(l=0, r=20, t=10, b=0),
                xaxis_title="Receita (Revenue em USD)", yaxis_title="Probabilidade Prevista (P)",
                yaxis=dict(range=[-0.05, 1.05], tickformat='.0%'),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )

            df_sens_sim = pd.concat([criar_vetor(revenue, g_teste, is_en) for g_teste in lista_generos], ignore_index=True)
            probs_sens = model_lgbm_nicho.predict_proba(df_sens_sim)[:, 1]
            
            df_sens = pd.DataFrame({'Classe': lista_generos, 'P(Sucesso)': probs_sens}).sort_values('P(Sucesso)', ascending=True)
            
            cores = ['#de2d26' if g == genre else '#bdbdbd' for g in df_sens['Classe']]
            fig_sens = px.bar(df_sens, x='P(Sucesso)', y='Classe', orientation='h', color='Classe', color_discrete_sequence=cores)

            fig_sens.add_vline(x=0.508, line_width=2, line_dash="dash", line_color="#de2d26", annotation_text="Limiar de Pareto - Top 20% (0.508)")
            
            fig_sens.update_layout(
                template=TEMPLATE, margin=dict(l=0, r=20, t=10, b=0), showlegend=False,
                xaxis_title="Probabilidade (P)", yaxis_title="",
                xaxis=dict(range=[0, 1.05], tickformat='.0%')
            )
            fig_sens.update_traces(marker_line_width=0)

            return fig_pdp, fig_sens, fig_imp, ""

        except Exception as e:
            return dash.no_update, dash.no_update, dash.no_update, dbc.Alert(f"Erro no processamento matricial: {str(e)}", color="danger")