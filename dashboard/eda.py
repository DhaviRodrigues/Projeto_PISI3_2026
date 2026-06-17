import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# --- Funções Estatísticas Auxiliares ---

def calcular_coeficiente_gini(y):
    """
    Calcula o Coeficiente de Gini corrigido para amostras finitas.
    Fórmula discreta baseada nas frações populacionais e acumuladas.
    """
    y_arr = np.asarray(y)
    y_arr = y_arr[y_arr >= 0]
    if len(y_arr) == 0 or y_arr.sum() == 0:
        return 0.0
    y_sorted = np.sort(y_arr)
    n = len(y_sorted)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * y_sorted) / (n * np.sum(y_sorted))) - (n + 1) / n
    if n > 1:
        gini = gini * (n / (n - 1))
    return float(gini)

def obter_curva_lorenz(y, max_pontos=1000):
    """
    Gera as coordenadas X e Y da Curva de Lorenz.
    """
    y_arr = np.asarray(y)
    y_arr = y_arr[y_arr >= 0]
    if len(y_arr) == 0 or y_arr.sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    
    y_sorted = np.sort(y_arr)
    cum_y = np.cumsum(y_sorted)
    lorenz_y = cum_y / cum_y[-1]
    
    # Inserir o ponto inicial (0,0)
    lorenz_y = np.insert(lorenz_y, 0, 0.0)
    lorenz_x = np.linspace(0, 1, len(lorenz_y))
    
    if len(lorenz_y) > max_pontos:
        indices = np.linspace(0, len(lorenz_y) - 1, max_pontos, dtype=int)
        lorenz_x = lorenz_x[indices]
        lorenz_y = lorenz_y[indices]
        
    return lorenz_x, lorenz_y

# --- Preparação e Limpeza de Dados ---

def preparar_dados(df):
    df_work = df.copy()
    
    if 'release_year' not in df_work.columns:
        df_work['release_year'] = pd.to_datetime(df_work.get('release_date', pd.Series(dtype='datetime64[ns]')), errors='coerce').dt.year

    if 'primeiro_genero' not in df_work.columns:
        df_work['primeiro_genero'] = (
            df_work['genres']
            .astype(str)
            .str.replace(r"[\[\]\'\"]", "", regex=True)
            .str.split(',')
            .str[0]
            .str.strip()
        )
        df_work['primeiro_genero'] = df_work['primeiro_genero'].replace({'nan': 'Desconhecido', 'None': 'Desconhecido', '': 'Desconhecido'})

    if 'primeiro_pais' not in df_work.columns:
        df_work['primeiro_pais'] = (
            df_work['production_countries']
            .astype(str)
            .str.replace(r"[\[\]\'\"]", "", regex=True)
            .str.split(',')
            .str[0]
            .str.strip()
        )
        df_work['primeiro_pais'] = df_work['primeiro_pais'].replace({'nan': 'Desconhecido', 'None': 'Desconhecido', '': 'Desconhecido'})

    if 'genres_list' not in df_work.columns:
        def _clean_genres(raw):
            items = [item.strip() for item in str(raw).replace("[", "").replace("]", "").replace("'", "").replace('"', '').split(',')]
            return [item for item in items if item and item.lower() not in ('nan', 'none')]

        df_work['genres_list'] = df_work['genres'].apply(_clean_genres)

    if 'vote_average_imputed' not in df_work.columns:
        mean_vote = df_work['vote_average'].dropna().mean()
        if np.isnan(mean_vote):
            mean_vote = 0.0
        df_work['vote_average_imputed'] = df_work['vote_average'].fillna(mean_vote)
        df_work.loc[df_work['vote_count'] == 0, 'vote_average_imputed'] = mean_vote

    return df_work

# --- Definição do Layout Dash ---

def create_eda_layout(df_preparado):
    if 'release_year' not in df_preparado.columns:
        df_preparado = preparar_dados(df_preparado)

    df_anos = df_preparado.dropna(subset=['release_year'])
    min_ano = int(df_anos['release_year'].min())
    max_ano = int(df_anos['release_year'].max())
    
    ano_inicio_padrao = max(min_ano, 2000)
    ano_fim_padrao = min(max_ano, 2030)

    generos_disponiveis = sorted(df_preparado[df_preparado['primeiro_genero'] != 'Desconhecido']['primeiro_genero'].dropna().unique())
    
    return html.Div([

        # --- CABEÇALHO ---
        html.H1("Análise Exploratória de Dados (EDA)", className="mb-2 text-dark font-weight-bold"),
        html.H3("Assimetria Distribucional e Teoria da Cauda Longa (Head/Tail Breaks)", className="mb-4 text-secondary"),
        
        html.P([
            "Exploração visual da assimetria na distribuição de receita, votos e produção do mercado cinematográfico, "
            "com partições baseadas no método estatístico Head/Tail Breaks."
        ], className="text-muted mb-4"),

        # --- FILTROS ---
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Intervalo de Anos:", className="fw-bold text-dark"),
                        html.Div([
                            dcc.RangeSlider(
                                id='filtro-ano',
                                min=min_ano, max=max_ano, step=1, marks=None, 
                                value=[ano_inicio_padrao, ano_fim_padrao], 
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'paddingTop': '10px', 'paddingBottom': '15px'})
                    ], width=7),
                    dbc.Col([
                        html.Label("Filtrar por Gênero Principal:", className="fw-bold text-dark"),
                        dcc.Dropdown(
                            id='filtro-genero',
                            options=[{'label': gen, 'value': gen} for gen in generos_disponiveis],
                            multi=True, placeholder="Selecione os gêneros..."
                        )
                    ], width=5),
                ])
            ])
        ], className="mb-5 shadow-sm border-0"),

        # --- INSIGHTS ---
        html.Div(id='insights-resumo', className='mb-4'),

        # --- SEÇÃO 1: ENGAJAMENTO (VOTOS) ---
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Checklist(
                        id='toggle-media-votos',
                        options=[{'label': ' Exibir Linhas de Ruptura (Média e Corte)', 'value': 'mostrar'}],
                        value=['mostrar'],
                        inline=True,
                        style={'textAlign': 'center', 'marginBottom': '10px'}
                    ),
                ]),
                dcc.Loading(dcc.Graph(id='grafico-ht-votos'))
            ], width=12)
        ], className="mb-5"),

        # --- SEÇÃO 2: RECEITA E LORENZ ---
        dbc.Row([
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-ht-receita'))
            ], width=7),
            
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-ht-comparativo'))
            ], width=5)
        ], className="mb-5"),

        # --- SEÇÃO 2.5: EVOLUÇÃO TEMPORAL DA CONCENTRAÇÃO (GINI POR ANO) ---
        dbc.Row([
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-gini-temporal'))
            ], width=12)
        ], className="mb-5"),

        # --- SEÇÃO 3: MONOPÓLIO GEOGRÁFICO ---
        dbc.Row([
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-geo-monopolio'))
            ], width=12)
        ], className="mb-5"),

        # --- SEÇÃO 4: GÊNEROS ---
        dbc.Row([
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-gen-mainstream'))
            ], width=6),
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-gen-tail'))
            ], width=6)
        ], className="mb-5"),

        # --- SEÇÃO 5: QUALIDADE (BOXPLOT) ---
        dbc.Row([
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-ht-qualidade'))
            ], width=12)
        ], className="mb-5"),

    ], style={"marginLeft": "12rem", "marginRight": "2rem", "paddingTop": "1rem"})


# --- Registro de Callbacks ---

def register_eda_callbacks(app, df_preparado):
    if 'release_year' not in df_preparado.columns:
        df_preparado = preparar_dados(df_preparado)

    @app.callback(
        Output('grafico-ht-votos', 'figure'),
        Input('toggle-media-votos', 'value')
    )
    def update_grafico_votos(mostrar_media_votos):
        df_sorted_votos = df_preparado.sort_values(by='vote_count', ascending=False).reset_index(drop=True)
        df_votes = df_sorted_votos[df_sorted_votos['vote_count'] > 0]
        
        if not df_votes.empty:
            media_v = df_votes['vote_count'].mean()
            corte_index = len(df_votes[df_votes['vote_count'] > media_v])
            
            df_head = df_votes.iloc[:corte_index]
            df_tail = df_votes.iloc[corte_index:]
            
            if len(df_tail) > 5000:
                df_tail = df_tail.iloc[::max(1, len(df_tail)//5000)]
            
            fig_ht_votos = go.Figure()
            fig_ht_votos.add_trace(go.Scatter(
                x=df_tail.index, y=df_tail['vote_count'], 
                mode='lines', fill='tozeroy', 
                fillcolor='rgba(217, 119, 6, 0.5)', 
                line=dict(color='#D97706', width=2), 
                name=f'Cauda Longa ({((len(df_votes)-corte_index)/len(df_votes))*100:.1f}%)'
            ))
            fig_ht_votos.add_trace(go.Scatter(
                x=df_head.index, y=df_head['vote_count'], 
                mode='lines', fill='tozeroy', 
                fillcolor='rgba(136, 19, 55, 0.7)', 
                line=dict(color='#881337', width=2), 
                name=f'Cabeça ({(corte_index/len(df_votes))*100:.1f}%)'
            ))
            
            fig_ht_votos.update_yaxes(type="log", title_text="Votos por Filme (Log)")
            fig_ht_votos.update_xaxes(title_text="Filmes (Ordenados por Engajamento)")
            
            if 'mostrar' in mostrar_media_votos:
                max_x = len(df_votes)
                max_y = df_votes['vote_count'].max()
                min_y = df_votes['vote_count'].min()
                fig_ht_votos.add_trace(go.Scatter(x=[0, max_x], y=[media_v, media_v], mode='lines', line=dict(color='#64748B', width=2, dash='dash'), name=f"Média Geral ({int(media_v)} votos)", hoverinfo='skip'))
                fig_ht_votos.add_trace(go.Scatter(x=[corte_index, corte_index], y=[min_y, max_y], mode='lines', line=dict(color='#475569', width=2, dash='dot'), name=f"Ponto de Ruptura ({corte_index:,}º filme)", hoverinfo='skip'))
                
            fig_ht_votos.update_layout(
                title="<b>Distribuição de Votos (Escala Logarítmica - Divisão Head/Tail Breaks)</b>",
                margin=dict(t=40, l=10, r=10, b=10), 
                hovermode='x unified', 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return fig_ht_votos
        return go.Figure()

    @app.callback(
        [Output('grafico-ht-receita', 'figure'),
         Output('grafico-ht-comparativo', 'figure'),
         Output('grafico-gini-temporal', 'figure'),
         Output('grafico-ht-qualidade', 'figure'),
         Output('grafico-geo-monopolio', 'figure'),     
         Output('grafico-gen-mainstream', 'figure'),    
         Output('grafico-gen-tail', 'figure'),
         Output('insights-resumo', 'children')],         
        [Input('filtro-ano', 'value'),
         Input('filtro-genero', 'value')] 
    )
    def update_graficos_filtrados(anos_selecionados, generos_selecionados):
        
        df_filtrado = df_preparado
        
        if anos_selecionados:
            ano_min, ano_max = anos_selecionados
            df_filtrado = df_filtrado[(df_filtrado['release_year'] >= ano_min) & (df_filtrado['release_year'] <= ano_max)]
            
        if generos_selecionados:
            df_filtrado = df_filtrado[df_filtrado['primeiro_genero'].isin(generos_selecionados)]
            
        vazio = go.Figure().update_layout(title="Sem dados para estes filtros")
        if df_filtrado.empty:
            info_vazio = dbc.Card(
                dbc.CardBody([
                    html.H5("Sem dados para estes filtros"),
                    html.P("Ajuste o intervalo de anos ou escolha outros gêneros para visualizar insights relevantes.", className="mb-0 text-muted small")
                ]),
                className="mb-4 shadow-sm border-0",
                style={"backgroundColor": "#f8fafc"}
            )
            return vazio, vazio, vazio, vazio, vazio, vazio, vazio, info_vazio

        df_rev = df_filtrado[df_filtrado['revenue'] > 0].sort_values(by='revenue', ascending=True).reset_index(drop=True)
        if not df_rev.empty:
            # 1. Curva de Lorenz e Coeficiente de Gini
            gini_val = calcular_coeficiente_gini(df_rev['revenue'])
            lorenz_x, lorenz_y = obter_curva_lorenz(df_rev['revenue'])
            
            fig_ht_rev = go.Figure()
            fig_ht_rev.add_trace(go.Scatter(
                x=lorenz_x, y=lorenz_y, 
                mode='lines', 
                line=dict(color='#881337', width=3), 
                fill='tozeroy',
                fillcolor='rgba(217, 119, 6, 0.15)',
                name=f'Curva de Lorenz (Gini = {gini_val:.3f})'
            ))
            fig_ht_rev.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], 
                mode='lines', 
                line=dict(color='#64748B', width=2, dash='dash'), 
                name='Perfeita Igualdade (Gini = 0.0)'
            ))
            
            fig_ht_rev.update_layout(
                title=f"<b>Curva de Lorenz (Concentração de Receita) - Gini: {gini_val:.3f}</b>",
                margin=dict(t=40, l=10, r=10, b=10),
                xaxis_title="Fração Acumulada de Filmes",
                yaxis_title="Fração Acumulada de Receita",
                xaxis=dict(range=[0, 1.02]),
                yaxis=dict(range=[0, 1.02]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # 2. Comparativo Head vs Tail (Volume vs Receita Total)
            media_receita = df_rev['revenue'].mean()
            df_rev['Grupo'] = np.where(df_rev['revenue'] > media_receita, 'Head', 'Tail')
            
            total_filmes = len(df_rev)
            filmes_head = len(df_rev[df_rev['Grupo'] == 'Head'])
            pct_filmes_head = (filmes_head / total_filmes) * 100
            pct_filmes_tail = 100 - pct_filmes_head
            
            receita_head = df_rev[df_rev['Grupo'] == 'Head']['revenue'].sum()
            pct_receita_head = (receita_head / df_rev['revenue'].sum()) * 100
            pct_receita_tail = 100 - pct_receita_head
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name='Tail (Cauda)', 
                x=['Volume Físico', 'Receita Total'], 
                y=[pct_filmes_tail, pct_receita_tail], 
                marker_color='#D97706', 
                text=[f"{pct_filmes_tail:.1f}%", f"{pct_receita_tail:.1f}%"], 
                textposition='inside'
            ))
            fig_comp.add_trace(go.Bar(
                name='Head (Cabeça)', 
                x=['Volume Físico', 'Receita Total'], 
                y=[pct_filmes_head, pct_receita_head], 
                marker_color='#881337', 
                text=[f"{pct_filmes_head:.1f}%", f"{pct_receita_head:.1f}%"], 
                textposition='inside'
            ))
            fig_comp.update_layout(
                title="<b>Composição do Mercado: Volume de Títulos vs. Receita Acumulada</b>",
                barmode='stack', 
                yaxis=dict(title='Porcentagem (%)', range=[0, 100]), 
                margin=dict(t=40, l=10, r=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
        else:
            fig_ht_rev = go.Figure()
            fig_comp = go.Figure()

        # 2.5. Evolução Histórica da Concentração (Gini por Ano)
        df_gini_temp = df_filtrado[df_filtrado['revenue'] > 0]
        if not df_gini_temp.empty:
            gini_por_ano = []
            anos_unicos = sorted(df_gini_temp['release_year'].dropna().unique())
            for ano in anos_unicos:
                receitas_ano = df_gini_temp[df_gini_temp['release_year'] == ano]['revenue']
                if len(receitas_ano) >= 5:  # Filtro de significância amostral
                    gini_val_ano = calcular_coeficiente_gini(receitas_ano)
                    gini_por_ano.append({'Ano': int(ano), 'Gini': gini_val_ano})
            
            if gini_por_ano:
                df_gini_plot = pd.DataFrame(gini_por_ano)
                fig_gini_temp = go.Figure()
                fig_gini_temp.add_trace(go.Scatter(
                    x=df_gini_plot['Ano'], 
                    y=df_gini_plot['Gini'], 
                    mode='lines+markers', 
                    line=dict(color='#881337', width=3), 
                    marker=dict(size=8, color='#881337'),
                    name='Gini por Ano'
                ))
                fig_gini_temp.update_layout(
                    title="<b>Evolução Histórica da Concentração de Receita (Coeficiente de Gini por Ano de Lançamento)</b>",
                    margin=dict(t=40, l=10, r=10, b=10),
                    xaxis_title="Ano de Lançamento",
                    yaxis_title="Coeficiente de Gini",
                    yaxis=dict(range=[0, 1.05]),
                    plot_bgcolor="#f8fafc"
                )
            else:
                fig_gini_temp = go.Figure().update_layout(title="Dados insuficientes para cálculo de Gini temporal")
        else:
            fig_gini_temp = go.Figure().update_layout(title="Sem dados de receita válidos para histórico temporal")

        # 3. Qualidade Percebida Boxplot
        df_boxplot = df_filtrado.sort_values(by='vote_count', ascending=False).reset_index(drop=True)
        df_votes_filt = df_boxplot[df_boxplot['vote_count'] > 0].copy()
        
        if not df_votes_filt.empty:
            media_v_filt = df_votes_filt['vote_count'].mean()
            corte_index_filt = len(df_votes_filt[df_votes_filt['vote_count'] > media_v_filt])
            
            df_boxplot['Segmento'] = np.where(df_boxplot.index < corte_index_filt, 'Mainstream (Head)', 'Alternativo (Tail)')
            
            fig_qualidade = px.box(
                df_boxplot, x='Segmento', y='vote_average_imputed', color='Segmento',
                color_discrete_map={'Mainstream (Head)': '#881337', 'Alternativo (Tail)': '#D97706'},
                labels={'vote_average_imputed': 'Nota Média', 'Segmento': 'Segmento'}
            )
            fig_qualidade.update_layout(
                title="<b>Comparação de Qualidade: Nota Média (vote_average) por Segmento</b>",
                yaxis=dict(range=[0, 10], title='Nota Média (0-10)'), 
                margin=dict(t=40, l=10, r=10, b=10), 
                showlegend=False
            )
        else:
            fig_qualidade = go.Figure()

        # 4. Monopólio Geográfico
        media_votos_filtrado = df_filtrado['vote_count'].mean()
        df_geo = df_filtrado[df_filtrado['primeiro_pais'] != 'Desconhecido']
        top_8_countries = df_geo['primeiro_pais'].value_counts().head(8).index.tolist()
        df_geo_filtered = df_geo[df_geo['primeiro_pais'].isin(top_8_countries)].copy()

        if not df_geo_filtered.empty:
            df_geo_filtered['Segmento'] = np.where(df_geo_filtered['vote_count'] >= media_votos_filtrado, 'Cabeça', 'Cauda')
            segment_counts = df_geo_filtered['Segmento'].value_counts()
            country_segment_counts = df_geo_filtered.groupby(['Segmento', 'primeiro_pais']).size().reset_index(name='count')
            
            country_segment_counts['percentage'] = country_segment_counts['count'] / country_segment_counts['Segmento'].map(segment_counts) * 100

            fig_geo = px.bar(
                country_segment_counts, x='primeiro_pais', y='percentage', color='Segmento', barmode='group',
                color_discrete_map={'Cabeça': '#881337', 'Cauda': '#D97706'},
                text=country_segment_counts['percentage'].apply(lambda x: f'{x:.1f}%')
            )
            fig_geo.update_traces(textposition='outside')
            fig_geo.update_layout(
                title="<b>Monopólio Geográfico: Origem dos Filmes por Segmento (Cabeça vs. Cauda)</b>",
                yaxis_title='Proporção (%)', 
                xaxis_title='País', 
                margin=dict(t=40, l=10, r=10, b=10), 
                legend_title_text=''
            )
        else:
            fig_geo = go.Figure()

        # 5. Distribuição de Gêneros (Rankings)
        df_mainstream = df_filtrado[df_filtrado['vote_count'] >= media_votos_filtrado]
        df_tail = df_filtrado[(df_filtrado['vote_count'] < media_votos_filtrado) & (df_filtrado['vote_count'] > 0)]

        df_gen_main = df_mainstream[df_mainstream['primeiro_genero'] != 'Desconhecido']['primeiro_genero'].value_counts().head(10).reset_index()
        df_gen_main.columns = ['Genre', 'Count']
        df_gen_main = df_gen_main.sort_values(by='Count', ascending=True) 

        df_gen_tail = df_tail[df_tail['primeiro_genero'] != 'Desconhecido']['primeiro_genero'].value_counts().head(10).reset_index()
        df_gen_tail.columns = ['Genre', 'Count']
        df_gen_tail = df_gen_tail.sort_values(by='Count', ascending=True)

        if not df_gen_main.empty and not df_gen_tail.empty:
            max_count = max(df_gen_main['Count'].max(), df_gen_tail['Count'].max())

            fig_gen_main = px.bar(df_gen_main, x='Count', y='Genre', orientation='h')
            fig_gen_main.update_traces(marker_color='#881337')
            fig_gen_main.update_layout(title=f"<b>Gêneros Mainstream (Cabeça - Votos > {int(media_votos_filtrado)})</b>", xaxis_range=[0, max_count], margin=dict(t=40, l=10, r=10, b=10))

            fig_gen_tail = px.bar(df_gen_tail, x='Count', y='Genre', orientation='h')
            fig_gen_tail.update_traces(marker_color='#D97706')
            fig_gen_tail.update_layout(title=f"<b>Gêneros Alternativos (Cauda - Votos <= {int(media_votos_filtrado)})</b>", xaxis_range=[0, max_count], margin=dict(t=40, l=10, r=10, b=10))
        else:
            fig_gen_main = go.Figure()
            fig_gen_tail = go.Figure()

        # 6. Sumário Analítico de Metadados
        top_head_genre = df_gen_main['Genre'].iloc[-1] if not df_gen_main.empty else 'N/A'
        top_tail_genre = df_gen_tail['Genre'].iloc[-1] if not df_gen_tail.empty else 'N/A'
        top_country = df_geo_filtered['primeiro_pais'].value_counts().idxmax() if not df_geo_filtered.empty else 'N/A'
        total_filmes = len(df_filtrado)
        pct_filmes_head = pct_filmes_head if 'pct_filmes_head' in locals() else 0.0
        pct_receita_head = pct_receita_head if 'pct_receita_head' in locals() else 0.0

        insights_children = dbc.Card(
            dbc.CardBody([
                html.H5("Metadados Estatísticos do Recorte Atual", className="card-title text-dark fw-bold"),
                html.P(f"A subamostra selecionada compreende {total_filmes:,} filmes distribuídos entre os anos de {anos_selecionados[0]} e {anos_selecionados[1]}.", className="mb-2 text-muted"),
                html.Ul([
                    html.Li([html.Strong("Concentração Comercial: "), f"A Cabeça do mercado captura {pct_receita_head:.1f}% de toda a receita acumulada, apesar de representar apenas {pct_filmes_head:.1f}% da população total de filmes com faturamento registrado."]),
                    html.Li([html.Strong("Hegemonia Geográfica: "), f"O país de produção com maior incidência no segmento de atenção crítica (Cabeça) é {top_country}."]),
                    html.Li([html.Strong("Gêneros de Destaque: "), f"O gênero dominante no mainstream é '{top_head_genre}', ao passo que na Cauda Longa destaca-se '{top_tail_genre}'."])
                ], className="mb-0 text-muted"),
            ]),
            className="mb-4 shadow-sm border-0",
            style={"backgroundColor": "#f8fafc"}
        )

        return fig_ht_rev, fig_comp, fig_gini_temp, fig_qualidade, fig_geo, fig_gen_main, fig_gen_tail, insights_children