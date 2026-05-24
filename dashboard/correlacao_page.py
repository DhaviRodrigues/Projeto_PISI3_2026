import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import scipy.stats as ss

# Dicionário mapeando os nomes amigáveis para as colunas do seu DataFrame
OPCOES_CORRELACAO = [
    {"label": "Receita", "value": "revenue"},
    {"label": "Orçamento", "value": "budget"},
    {"label": "Duração", "value": "runtime"},
    {"label": "Nota Média", "value": "vote_average"},
    {"label": "Total de Votos", "value": "vote_count"},
    {"label": "Popularidade", "value": "popularity"}
]

OPCOES_LINHA = [
    {"label": "Receita", "value": "revenue"},
    {"label": "Orçamento", "value": "budget"},
    {"label": "Duração", "value": "runtime"},
    {"label": "Nota Média", "value": "vote_average"},
    {"label": "Total de Votos", "value": "vote_count"},
    {"label": "Popularidade", "value": "popularity"}
]

def create_correlation_layout():
    """Gera o layout da página de correlações"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Correlação Entre Dados", className="mb-2"),
                html.H2("Matriz de Correlação Interativa", className="mb-2"),
                html.P(
                    "Descubra se as variáveis sobem ou descem juntas. Valores próximos a 1 "
                    "indicam que aumentam juntas. Valores próximos a -1 indicam o inverso.",
                    className="text-muted"
                ),
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Selecione as variáveis para comparar:", className="fw-bold"),
                dcc.Dropdown(
                    id="dropdown-colunas-correlacao",
                    options=OPCOES_CORRELACAO,
                    # Removido 'is_original_en' do padrão inicial
                    value=["budget", "revenue", "popularity", "vote_average"], 
                    multi=True,
                    placeholder="Selecione duas ou mais colunas...",
                    className="mb-4"
                )
            ], width=8),
            dbc.Col([
                html.Label("Filtro de Dados (Matriz):", className="fw-bold mb-2"),
                dbc.Checklist(
                    options=[
                        {"label": " Ocultar valores zerados ou nulos (0 e NaN)", "value": "ativado"}
                    ],
                    value=["ativado"], 
                    id="checkbox-filtro-zeros-matriz",
                    className="mb-4"
                )
            ], width=4)
        ]),

        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-correlacao",
                            type="default",
                            color="#0d6efd",
                            children=dcc.Graph(id="grafico-correlacao", style={"height": "600px"})
                        )
                    ])
                ], outline=True, color="primary", className="shadow-sm")
            ], width=12)
        ]),
        
        html.Hr(className="my-5"), # Linha divisória
        
        dbc.Row([
            dbc.Col([
                html.H2("Curva de Impacto e Crescimento", className="mb-2"),
                html.P("Veja como o comportamento médio da variável Y muda conforme a variável X cresce.", className="text-muted"),
            ], width=12)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Variável Base (Eixo X):", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id="dropdown-linha-x",
                    options=OPCOES_LINHA,
                    value="budget", 
                    clearable=False,
                    className="mb-3"
                )
            ], width=4), # Reduzido para width 4
            
            dbc.Col([
                html.Label("Variável de Impacto (Eixo Y):", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id="dropdown-linha-y",
                    options=OPCOES_LINHA,
                    value="revenue", 
                    clearable=False,
                    className="mb-3"
                )
            ], width=4), # Reduzido para width 4
            
            dbc.Col([
                html.Label("Filtro de Dados:", className="fw-bold mb-2"),
                dbc.Checklist(
                    options=[
                        {"label": " Ocultar valores zerados ou nulos (0 e NaN)", "value": "ativado"}
                    ],
                    # Inicia com o filtro ligado por padrão para exibir a curva correta
                    value=["ativado"], 
                    id="checkbox-filtro-zeros",
                    className="mb-3"
                )
            ], width=4) # Ocupa os 4 espaços restantes
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-linha",
                            type="default",
                            color="#3a86ff",
                            children=dcc.Graph(id="grafico-linha-impacto", style={"height": "500px"})
                        )
                    ])
                ], outline=True, color="info", className="shadow-sm mb-5")
            ], width=12)
        ])

    ], style={"marginLeft": "9rem", "marginRight": "2rem", "paddingTop": "1rem"})

def calcular_cramer_v(coluna1, coluna2):
    """Calcula a associação V de Cramer entre duas variáveis categóricas."""
    # Cria a tabela de contingência (frequência cruzada) removendo nulos temporários
    tabela_contingencia = pd.crosstab(coluna1, coluna2)
    
    # Se a tabela for irrelevante (ex: apenas 1 categoria), retorna 0
    if tabela_contingencia.size == 0 or min(tabela_contingencia.shape) <= 1:
        return 0.0
        
    # Calcula o Qui-Quadrado
    chi2 = ss.chi2_contingency(tabela_contingencia)[0]
    n = tabela_contingencia.sum().sum()
    
    # Dimensões da tabela
    r, k = tabela_contingencia.shape
    
    # Aplica a fórmula do V de Cramer
    return np.sqrt(chi2 / (n * min(k - 1, r - 1)))

def register_correlation_callbacks(app, df):
    """
    Registra os callbacks. 
    Nota: Você precisa passar o seu DataFrame (df) principal quando chamar esta função.
    """
    
    @app.callback(
        Output("grafico-linha-impacto", "figure"),
        [
            Input("dropdown-linha-x", "value"),
            Input("dropdown-linha-y", "value"),
            Input("checkbox-filtro-zeros", "value") # Novo input do checkbox
        ]
    )
    def atualizar_grafico_linha_impacto(coluna_x, coluna_y, filtro_checkbox):
        if not coluna_x or not coluna_y or coluna_x == coluna_y:
            return go.Figure()

        df_plot = df.copy()

        # [REMOVIDO]: O bloco antigo 'for col in [coluna_x, coluna_y]:' contendo is_original_en, etc, foi deletado.

        ignorar_zeros = filtro_checkbox and "ativado" in filtro_checkbox

        if ignorar_zeros:
            df_plot = df_plot.dropna(subset=[coluna_x, coluna_y])
            # Como todas as variáveis restantes são puramente numéricas, a validação fica direta:
            df_plot[coluna_x] = pd.to_numeric(df_plot[coluna_x], errors='coerce')
            df_plot[coluna_y] = pd.to_numeric(df_plot[coluna_y], errors='coerce')
            df_plot = df_plot[(df_plot[coluna_x] > 0) & (df_plot[coluna_y] > 0)]
        else:
            df_plot[coluna_x] = pd.to_numeric(df_plot[coluna_x], errors='coerce').fillna(0)
            df_plot[coluna_y] = pd.to_numeric(df_plot[coluna_y], errors='coerce').fillna(0)

        # Prevenção contra dataframe vazio após os filtros
        if df_plot.empty:
            return go.Figure()

        # 4. Engenharia de Agrupamento Dinâmica (Solução pd.cut aplicada)
        try:
            # Variáveis contínuas extensas
            if df_plot[coluna_x].nunique() > 15:
                # pd.cut cria fatias fixas do mínimo ao máximo (evita colapso de dados do antigo qcut)
                df_plot["faixa_x"] = pd.cut(df_plot[coluna_x], bins=12)
                
                df_agrupado = df_plot.groupby("faixa_x", as_index=False, observed=False).agg(
                    {coluna_x: "mean", coluna_y: "mean"}
                ).dropna().sort_values(by=coluna_x)
            else:
                # Discretas (ex: 0 e 1, quantidade de países)
                df_agrupado = df_plot.groupby(coluna_x, as_index=False).agg({coluna_y: "mean"}).sort_values(by=coluna_x)
        except:
            # Fallback seguro
            df_agrupado = df_plot.groupby(coluna_x, as_index=False).agg({coluna_y: "mean"}).sort_values(by=coluna_x)

        mapa_nomes = {item["value"]: item["label"] for item in OPCOES_LINHA}

        # 5. Cria o Gráfico de Linha de Tendência Limpo
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_agrupado[coluna_x],
            y=df_agrupado[coluna_y],
            mode="lines+markers",
            line=dict(color="#3a86ff", width=4),
            marker=dict(size=10, color="#3a86ff", symbol="circle"),
            name="Tendência Média"
        ))
        # 6. Configuração de Layout e Negrito
        fig.update_layout(
            title=f"<b>Curva de Crescimento: Como '{mapa_nomes[coluna_x]}' afeta '{mapa_nomes[coluna_y]}'</b>",
            xaxis_title=f"<b>{mapa_nomes[coluna_x]}</b>",
            yaxis_title=f"<b>{mapa_nomes[coluna_y]}</b>",
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            margin=dict(l=90, r=20, t=60, b=60),
            xaxis=dict(tickfont=dict(weight="bold", size=11)),
            yaxis=dict(tickfont=dict(weight="bold", size=11)),
            font=dict(weight="bold")
        )

        return fig
        
    @app.callback(
        Output("grafico-correlacao", "figure"),
        [
            Input("dropdown-colunas-correlacao", "value"),
            Input("checkbox-filtro-zeros-matriz", "value")
        ]
    )
    def atualizar_grafico_correlacao(colunas_selecionadas, filtros_checkbox):
        if not colunas_selecionadas or len(colunas_selecionadas) < 2:
            # Cria uma figura vazia sem dados (sem quadrados, sem barra de cores)
            fig = go.Figure()
            
            # Adiciona apenas o texto centralizado na tela
            fig.add_annotation(
                text="<b>Selecione pelo menos 2 variáveis para ver a correlação</b>",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, weight="bold", color="#495057")
            )
            
            # Limpa completamente o fundo, eixos e linhas para não exibir nada
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            return fig

        # 1. Copia o dataframe para não alterar o original em memória
        df_plot = df.copy()

        # [REMOVIDO]: Toda a Engenharia de Recursos Dinâmica (is_original_en, num_countries, etc.) foi deletada daqui.

        ignorar_zeros = filtros_checkbox and "ativado" in filtros_checkbox

        if ignorar_zeros:
            # Remove linhas com NaN nas colunas selecionadas
            df_plot = df_plot.dropna(subset=colunas_selecionadas)
            
            # Limpa valores zerados de forma universal para todas as colunas numéricas restantes
            for c in colunas_selecionadas:
                df_plot[c] = pd.to_numeric(df_plot[c], errors='coerce')
                df_plot = df_plot[df_plot[c] > 0]
        else:
            # Mantém todos os dados mesmo vazios, convertendo os NaNs para 0
            for c in colunas_selecionadas:
                df_plot[c] = pd.to_numeric(df_plot[c], errors='coerce').fillna(0)

        if df_plot.empty or len(df_plot) < 2:
            return go.Figure()
        
        # 4. Filtra apenas as colunas selecionadas para o cálculo numérico
        df_corr = df_plot[colunas_selecionadas].corr(numeric_only=True)

        # === ALTERAÇÃO: MÁSCARA DO TRIÂNGULO INFERIOR ===
        # Cria uma matriz booleana (True onde queremos ocultar: triângulo estritamente superior)
        mascara = np.triu(np.ones(df_corr.shape), k=1).astype(bool)
        # Substitui os valores repetidos do triângulo superior por NaN
        df_corr_mascarado = df_corr.where(~mascara)

        # Mapeamento para nomes amigáveis nos eixos
        mapa_nomes = {item["value"]: item["label"] for item in OPCOES_CORRELACAO}
        df_corr_mascarado.rename(index=mapa_nomes, columns=mapa_nomes, inplace=True)


        escala_customizada = [
            [0.0, "#fce4ec"],   # -1 (Rosa neutro desbotado)
            [0.5, "#f4f6f6"],   #  0 (Nula: Cinza muito claro)
            [0.75, "#90e0ef"],  #  0.5 (Moderada: Azul piscina)
            [1.0, "#3a86ff"]    #  1.0 (Forte Correlação: Azul Claro Vivo)
        ]

        fig = px.imshow(
            df_corr_mascarado, # Usa a matriz mascarada
            text_auto=".2f", 
            aspect="auto",
            color_continuous_scale=escala_customizada, 
            zmin=-1, 
            zmax=1,  
            labels={"color": "Correlação"}
        )
        
        fig.update_layout(
            title="<b>Mapa de Correlação de Pearson entre Colunas</b>",
            xaxis_title="",
            yaxis_title="",
            margin=dict(l=190, r=20, t=50, b=80),
            xaxis=dict(
                tickfont=dict(weight="bold", size=12),
                ticksuffix="   "
            ),
            yaxis=dict(
                tickfont=dict(weight="bold", size=12),
                ticksuffix="   "
            ),
            font=dict(weight="bold"),
            plot_bgcolor="white"
        )
        
        return fig
    
    @app.callback(
    Output("grafico-correlacao-categorica", "figure"),
    [
        Input("dropdown-categoricas", "value"),
        Input("checkbox-filtro-nulos-cat", "value")
    ])

    def atualizar_matriz_categorica(colunas_selecionadas, filtro_checkbox):
        if not colunas_selecionadas or len(colunas_selecionadas) < 2:
            return go.Figure() # Retorna figura vazia se tiver menos de 2

        df_plot = df.copy()
        
        # Inicializa uma matriz vazia com as colunas selecionadas
        n_colunas = len(colunas_selecionadas)
        matriz_v = np.zeros((n_colunas, n_colunas))
        
        # Preenche a matriz calculando par por par
        for i in range(n_colunas):
            for j in range(n_colunas):
                if i == j:
                    matriz_v[i, j] = 1.0  # Correlação de uma variável com ela mesma é sempre 1
                elif i > j:
                    # Remove nulos apenas do par específico para não estragar a amostra das outras colunas
                    df_par = df_plot[[colunas_selecionadas[i], colunas_selecionadas[j]]].dropna()
                    
                    v_val = calcular_cramer_v(df_par[colunas_selecionadas[i]], df_par[colunas_selecionadas[j]])
                    
                    matriz_v[i, j] = v_val
                    matriz_v[j, i] = v_val # A matriz é simétrica

        # Transforma de volta em DataFrame para o Plotly ler
        df_cramer = pd.DataFrame(matriz_v, index=colunas_selecionadas, columns=colunas_selecionadas)
        
        mascara = np.triu(np.ones(df_cramer.shape), k=1).astype(bool)
        df_cramer_mascarado = df_cramer.where(~mascara)
        
        # Configura uma escala de cores que começa no cinza (0.0) e vai para o Azul Claro (1.0)
        escala_cramer = [
            [0.0, "#f4f6f6"],  # Sem associação
            [0.3, "#90e0ef"],  # Associação fraca/moderada
            [1.0, "#3a86ff"]   # Associação forte (Azul Claro)
        ]
        
        fig = px.imshow(
            df_cramer_mascarado,
            text_auto=".2f",
            zmin=0, zmax=1, # O V de Cramer nunca é negativo!
            color_continuous_scale=escala_cramer,
            labels={"color": "V de Cramer"}
        )
        
        fig.update_layout(plot_bgcolor="white", font=dict(weight="bold"))
        return fig