import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import re


def extrair_primeiro_item(texto):
    """Função robusta para limpar colchetes, aspas e extrair o primeiro item de uma lista em string"""
    if pd.isna(texto) or str(texto).strip() in ['nan', 'None', '', '[]']:
        return 'Desconhecido'
 
    texto_limpo = re.sub(r"[\[\]\'\"]", "", str(texto))

    return texto_limpo.split(',')[0].strip()

def preparar_dados(df):
    df_work = df.copy()
    
    if 'release_year' not in df_work.columns:
        df_work['release_year'] = pd.to_datetime(df_work['release_date'], errors='coerce').dt.year

    if 'primeiro_genero' not in df_work.columns:
        df_work['primeiro_genero'] = df_work['genres'].apply(extrair_primeiro_item)
        
    if 'primeiro_pais' not in df_work.columns:
        df_work['primeiro_pais'] = df_work['production_countries'].apply(extrair_primeiro_item)

    return df_work


def create_eda_layout(df_cru):
    df = preparar_dados(df_cru)
    
    df_anos = df.dropna(subset=['release_year'])
    min_ano = int(df_anos['release_year'].min())
    max_ano = int(df_anos['release_year'].max())
    
    generos_disponiveis = sorted(df[df['primeiro_genero'] != 'Desconhecido']['primeiro_genero'].unique())
    
    return html.Div([
        html.H2("A Teoria da Cauda Longa (Head/Tail Breaks)", className="mb-4 text-primary"),
        html.P("Exploração visual da profunda desigualdade na distribuição de receita, votos e produção cinematográfica.", className="text-muted mb-4"),
        
        # --- FILTROS ---
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Intervalo de Anos:", className="fw-bold"),
                        dcc.RangeSlider(
                            id='filtro-ano',
                            min=min_ano, max=max_ano, step=1, marks=None, 
                            value=[min_ano, max_ano], 
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], width=7),
                    dbc.Col([
                        html.Label("Filtrar por Género:", className="fw-bold"),
                        dcc.Dropdown(
                            id='filtro-genero',
                            options=[{'label': gen, 'value': gen} for gen in generos_disponiveis],
                            multi=True, placeholder="Selecione os géneros..."
                        )
                    ], width=5),
                ])
            ])
        ], className="mb-5 shadow-sm"),

        dbc.Row([
            dbc.Col([
                html.H5("Curva de Desigualdade da Receita", className="text-center"),
                html.P("Como poucos filmes capturam quase todo o capital do mercado", className="text-center text-muted small"),
                dcc.Loading(dcc.Graph(id='grafico-ht-receita'))
            ], width=7),
            
            dbc.Col([
                html.H5("Comparativo: Head vs Tail", className="text-center"),
                html.P("Volume de Filmes vs Volume de Receita", className="text-center text-muted small"),
                dcc.Loading(dcc.Graph(id='grafico-ht-comparativo'))
            ], width=5)
        ], className="mb-5"),

        dbc.Row([
            dbc.Col([
                html.H5("Curva de Engajamento: Votos por Filme", className="text-center"),
                html.P("O abismo entre os blockbusters globais e os filmes locais/de nicho", className="text-center text-muted small"),
                html.Div([
                    dcc.Checklist(
                        id='toggle-media-votos',
                        options=[{'label': ' Exibir Linhas de Ruptura (Média)', 'value': 'mostrar'}],
                        value=['mostrar'],
                        inline=True,
                        style={'textAlign': 'center', 'marginBottom': '10px'}
                    ),
                ]),
                dcc.Loading(dcc.Graph(id='grafico-ht-votos'))
            ], width=12)
        ], className="mb-5"),
        
        dbc.Row([
            dbc.Col([
                html.H5("Qualidade Percebida: Head vs. Tail", className="text-center"),
                html.P("Embora a 'Head' domine a visibilidade, as notas são estatisticamente superiores? ""(C/ Imputação Proporcional)", className="text-center text-muted small"),
                dcc.Loading(dcc.Graph(id='grafico-ht-qualidade'))
            ], width=12)
        ], className="mb-5"),

        
        dbc.Row([
            dbc.Col([
                html.H5("Produção por Países: Mainstream vs Cinema Alternativo", className="text-center mb-3"),
                
                html.Div([
                    dbc.RadioItems(
                        id="toggle-paises",
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=[
                            {"label": " Visualizar Top 10 da Head (Mainstream)", "value": "Head"},
                            {"label": " Visualizar Top 10 da Tail (Cinema Alternativo)", "value": "Tail"},
                        ],
                        value="Head", 
                    )
                ], className="d-flex justify-content-center mb-3"),
                
                dcc.Loading(dcc.Graph(id='grafico-ht-paises'))
            ], width=12)
        ], className="mb-5")
    ])


def register_eda_callbacks(app, df_cru):
    df = preparar_dados(df_cru)

    @app.callback(
        [Output('grafico-ht-receita', 'figure'),
         Output('grafico-ht-comparativo', 'figure'),
         Output('grafico-ht-votos', 'figure'),
         Output('grafico-ht-qualidade', 'figure'),
         Output('grafico-ht-paises', 'figure')],
         
        [Input('filtro-ano', 'value'),
         Input('filtro-genero', 'value'),
         Input('toggle-paises', 'value'),
         Input('toggle-media-votos', 'value')] 
    )
    def update_graficos(anos_selecionados, generos_selecionados, toggle_paises_val, mostrar_media_votos):
       
        df_filtrado = df.copy()
        
        if anos_selecionados:
            ano_min, ano_max = anos_selecionados
            df_filtrado = df_filtrado[(df_filtrado['release_year'] >= ano_min) & 
                                      (df_filtrado['release_year'] <= ano_max)]
            
        if generos_selecionados:
            pattern = '|'.join(generos_selecionados)
            df_filtrado = df_filtrado[df_filtrado['genres'].str.contains(pattern, case=False, na=False)]
            
        if df_filtrado.empty:
            vazio = go.Figure().update_layout(title="Sem dados para estes filtros")
            return vazio, vazio, vazio, vazio, vazio

       
        df_rev = df_filtrado[df_filtrado['revenue'] > 0].sort_values(by='revenue', ascending=False).reset_index(drop=True)
        
        if not df_rev.empty:
            media_receita = df_rev['revenue'].mean()
            df_rev['Grupo'] = np.where(df_rev['revenue'] > media_receita, 'Head', 'Tail')
            
            fig_ht_rev = px.area(
                df_rev, x=df_rev.index, y='revenue', color='Grupo',
                color_discrete_map={'Head': '#D90429', 'Tail': '#2B2D42'},
                labels={'index': 'Ranking de Filmes', 'revenue': 'Receita (USD)'}
            )
            fig_ht_rev.update_yaxes(type="linear", tickformat=".2s", range=[0, min(800000000, df_rev['revenue'].max())])
            fig_ht_rev.add_hline(y=media_receita, line_dash="dash", line_color="black", annotation_text="Média Matemática")
            fig_ht_rev.update_layout(margin=dict(t=20, l=10, r=10, b=10), showlegend=True)
        else:
            fig_ht_rev = go.Figure()

       
        if not df_rev.empty:
            total_filmes = len(df_rev)
            total_receita = df_rev['revenue'].sum()
            
            filmes_head = len(df_rev[df_rev['Grupo'] == 'Head'])
            receita_head = df_rev[df_rev['Grupo'] == 'Head']['revenue'].sum()
            
            pct_filmes_head = (filmes_head / total_filmes) * 100
            pct_filmes_tail = 100 - pct_filmes_head
            
            pct_receita_head = (receita_head / total_receita) * 100
            pct_receita_tail = 100 - pct_receita_head
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='Tail (Cauda)', x=['Volume de Filmes', 'Receita Total'], y=[pct_filmes_tail, pct_receita_tail], marker_color='#2B2D42', text=[f"{pct_filmes_tail:.1f}%", f"{pct_receita_tail:.1f}%"], textposition='inside'))
            fig_comp.add_trace(go.Bar(name='Head (Cabeça)', x=['Volume de Filmes', 'Receita Total'], y=[pct_filmes_head, pct_receita_head], marker_color='#D90429', text=[f"{pct_filmes_head:.1f}%", f"{pct_receita_head:.1f}%"], textposition='inside'))
            fig_comp.update_layout(barmode='stack', yaxis=dict(title='Porcentagem (%)', range=[0, 100]), margin=dict(t=20, l=10, r=10, b=10))
        else:
            fig_comp = go.Figure()

       
        df_sorted_votos = df_filtrado.sort_values(by='vote_count', ascending=False).reset_index(drop=True)
        df_votes = df_sorted_votos[df_sorted_votos['vote_count'] > 0].copy()
        
        if not df_votes.empty:
            media_v = df_votes['vote_count'].mean()
            corte_index = len(df_votes[df_votes['vote_count'] > media_v])
            
            x_head = df_votes.index[:corte_index]; y_head = df_votes['vote_count'].iloc[:corte_index]
            x_tail = df_votes.index[corte_index:]; y_tail = df_votes['vote_count'].iloc[corte_index:]
            
            fig_ht_votos = go.Figure()
            fig_ht_votos.add_trace(go.Scatter(x=x_tail, y=y_tail, mode='lines', fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.5)', line=dict(color='darkorange', width=2), name=f'Cauda Longa ({((len(df_votes)-corte_index)/len(df_votes))*100:.1f}%)'))
            fig_ht_votos.add_trace(go.Scatter(x=x_head, y=y_head, mode='lines', fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.7)', line=dict(color='royalblue', width=2), name=f'Cabeça ({(corte_index/len(df_votes))*100:.1f}%)'))
            fig_ht_votos.add_trace(go.Scatter(x=df_votes.index, y=df_votes['vote_count'], mode='lines', line=dict(color='black', width=1.5), showlegend=False, hoverinfo='skip'))

            fig_ht_votos.update_yaxes(type="log", title_text="Votos por Filme (Escala Log)")
            fig_ht_votos.update_xaxes(title_text="Quantidade de Filmes (Ranking)")
            
            if 'mostrar' in mostrar_media_votos:
                max_x = len(df_votes); max_y = df_votes['vote_count'].max(); min_y = df_votes['vote_count'].min()
                fig_ht_votos.add_trace(go.Scatter(x=[0, max_x], y=[media_v, media_v], mode='lines', line=dict(color='red', width=2, dash='dash'), name=f"Média Aritmética ({int(media_v)} votos)", hoverinfo='skip'))
                fig_ht_votos.add_trace(go.Scatter(x=[corte_index, corte_index], y=[min_y, max_y], mode='lines', line=dict(color='darkred', width=2, dash='dot'), name=f"Ruptura (Index: {corte_index:,})", hoverinfo='skip'))
                
            fig_ht_votos.update_layout(margin=dict(t=20, l=10, r=10, b=10), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            df_boxplot = df_sorted_votos.copy()
            notas_disponiveis = df_votes['vote_average'].values
            mask_zero = df_boxplot['vote_count'] == 0
            num_zeros = mask_zero.sum()
            df_boxplot['vote_average_imputed'] = df_boxplot['vote_average']
            
            if num_zeros > 0 and len(notas_disponiveis) > 0:
                notas_imputadas = np.random.choice(notas_disponiveis, size=num_zeros)
                df_boxplot.loc[mask_zero, 'vote_average_imputed'] = notas_imputadas
                
            df_boxplot['Segmento'] = np.where(df_boxplot.index < corte_index, 'Mainstream (Head)', 'Alternativo (Tail)')
            
            fig_qualidade = px.box(
                df_boxplot, x='Segmento', y='vote_average_imputed', color='Segmento',
                color_discrete_map={'Mainstream (Head)': 'royalblue', 'Alternativo (Tail)': 'darkorange'},
                labels={'vote_average_imputed': 'Nota Média Suavizada', 'Segmento': 'Segmento de Mercado'}
            )
            fig_qualidade.update_layout(yaxis=dict(range=[0, 10]), margin=dict(t=20, l=10, r=10, b=10), showlegend=False)

        else:
            fig_ht_votos = go.Figure()
            fig_qualidade = go.Figure()

    
        df_paises = df_filtrado['primeiro_pais'].value_counts().reset_index()
        df_paises.columns = ['País', 'Qtd_Filmes']
        df_paises = df_paises[df_paises['País'] != 'Desconhecido']
        
        if not df_paises.empty:
            media_paises = df_paises['Qtd_Filmes'].mean()
            
            if toggle_paises_val == 'Head':
               
                df_filtrado_paises = df_paises[df_paises['Qtd_Filmes'] > media_paises].copy()
                cor_barra = '#D90429' 
                titulo_grafico = "Top 10 Países no Segmento 'Head' (Mainstream)"
               
                df_final_paises = df_filtrado_paises.head(10).sort_values(by='Qtd_Filmes', ascending=True)
            else:
               
                df_filtrado_paises = df_paises[df_paises['Qtd_Filmes'] <= media_paises].copy()
                cor_barra = '#2B2D42' 
                titulo_grafico = "Top 10 Países no Segmento 'Tail' (Alternativo)"
               
                df_final_paises = df_filtrado_paises.sort_values(by='Qtd_Filmes', ascending=False).head(10).sort_values(by='Qtd_Filmes', ascending=True)

            fig_ht_paises = px.bar(
                df_final_paises, x='Qtd_Filmes', y='País', orientation='h', title=titulo_grafico
            )
            fig_ht_paises.update_traces(marker_color=cor_barra)
            fig_ht_paises.update_layout(
                margin=dict(t=40, l=10, r=10, b=10), 
                xaxis_title="Quantidade de Filmes Produzidos", 
                yaxis_title=""
            )
        else:
            fig_ht_paises = go.Figure()

        return fig_ht_rev, fig_comp, fig_ht_votos, fig_qualidade, fig_ht_paises