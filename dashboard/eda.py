import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

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

    return df_work

def create_eda_layout(df_preparado):
    if 'release_year' not in df_preparado.columns:
        df_preparado = preparar_dados(df_preparado)

    df_anos = df_preparado.dropna(subset=['release_year'])
    min_ano = int(df_anos['release_year'].min())
    max_ano = int(df_anos['release_year'].max())
    
    ano_inicio_padrao = max(min_ano, 2000)
    ano_fim_padrao = min(max_ano, 2030)
    
    generos_disponiveis = sorted(df_preparado[df_preparado['primeiro_genero'] != 'Desconhecido']['primeiro_genero'].unique())
    

    return html.Div([

        html.H2("A Teoria da Cauda Longa (Head/Tail Breaks)", className="mb-4 text-primary", style={'marginTop': '40px'}),
        html.P("Exploração visual da profunda desigualdade na distribuição de receita, votos e produção cinematográfica.", className="text-muted mb-4"),
        
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Intervalo de Anos:", className="fw-bold"),
                        # --- ESPAÇAMENTO INTERNO PARA O SLIDER NÃO CORTAR O TOOLTIP ---
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
                html.H5("Curva de Engajamento: Votos por Filme", className="text-center"),
                html.P("O abismo entre os blockbusters globais e os filmes locais/de nicho (Não afetado pelos filtros)", className="text-center text-muted small"),
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
                html.H5("O Monopólio Geográfico", className="text-center"),
                html.P("Origem dos Filmes por Segmento (Proporção dentro de cada mercado)", className="text-center text-muted small"),
                dcc.Loading(dcc.Graph(id='grafico-geo-monopolio'))
            ], width=12)
        ], className="mb-5"),

        dbc.Row([
            html.H5("Top 10 Gêneros: Mainstream vs. Alternativos", className="text-center mb-3"),
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-gen-mainstream'))
            ], width=6),
            dbc.Col([
                dcc.Loading(dcc.Graph(id='grafico-gen-tail'))
            ], width=6)
        ], className="mb-5"),

        dbc.Row([
            dbc.Col([
                html.H5("Qualidade Percebida: Head vs. Tail", className="text-center"),
                html.P("Embora a 'Head' domine a visibilidade, as notas são estatisticamente superiores? (C/ Imputação Proporcional)", className="text-center text-muted small"),
                dcc.Loading(dcc.Graph(id='grafico-ht-qualidade'))
            ], width=12)
        ], className="mb-5"),
    ], style={"marginLeft": "18rem", "marginRight": "2rem", "paddingTop": "1rem"})

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
            fig_ht_votos.add_trace(go.Scatter(x=df_tail.index, y=df_tail['vote_count'], mode='lines', fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.5)', line=dict(color='darkorange', width=2), name=f'Cauda Longa ({((len(df_votes)-corte_index)/len(df_votes))*100:.1f}%)'))
            fig_ht_votos.add_trace(go.Scatter(x=df_head.index, y=df_head['vote_count'], mode='lines', fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.7)', line=dict(color='royalblue', width=2), name=f'Cabeça ({(corte_index/len(df_votes))*100:.1f}%)'))
            
            fig_ht_votos.update_yaxes(type="log", title_text="Votos por Filme (Escala Log)")
            fig_ht_votos.update_xaxes(title_text="Quantidade de Filmes (Ranking)")
            
            if 'mostrar' in mostrar_media_votos:
                max_x = len(df_votes)
                max_y = df_votes['vote_count'].max()
                min_y = df_votes['vote_count'].min()
                fig_ht_votos.add_trace(go.Scatter(x=[0, max_x], y=[media_v, media_v], mode='lines', line=dict(color='red', width=2, dash='dash'), name=f"Média ({int(media_v)} votos)", hoverinfo='skip'))
                fig_ht_votos.add_trace(go.Scatter(x=[corte_index, corte_index], y=[min_y, max_y], mode='lines', line=dict(color='darkred', width=2, dash='dot'), name=f"Ruptura ({corte_index:,})", hoverinfo='skip'))
                
            fig_ht_votos.update_layout(margin=dict(t=20, l=10, r=10, b=10), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            return fig_ht_votos
        return go.Figure()

    @app.callback(
        [Output('grafico-ht-receita', 'figure'),
         Output('grafico-ht-comparativo', 'figure'),
         Output('grafico-ht-qualidade', 'figure'),
         Output('grafico-geo-monopolio', 'figure'),     
         Output('grafico-gen-mainstream', 'figure'),    
         Output('grafico-gen-tail', 'figure')],         
        [Input('filtro-ano', 'value'),
         Input('filtro-genero', 'value')] 
    )
    def update_graficos_filtrados(anos_selecionados, generos_selecionados):
        
        df_filtrado = df_preparado
        
        if anos_selecionados:
            ano_min, ano_max = anos_selecionados
            df_filtrado = df_filtrado[(df_filtrado['release_year'] >= ano_min) & (df_filtrado['release_year'] <= ano_max)]
            
        if generos_selecionados:
            pattern = '|'.join(generos_selecionados)
            df_filtrado = df_filtrado[df_filtrado['genres'].str.contains(pattern, case=False, na=False)]
            
        vazio = go.Figure().update_layout(title="Sem dados para estes filtros")
        if df_filtrado.empty:
            return vazio, vazio, vazio, vazio, vazio, vazio

        df_rev = df_filtrado[df_filtrado['revenue'] > 0].sort_values(by='revenue', ascending=False).reset_index(drop=True)
        if not df_rev.empty:
            media_receita = df_rev['revenue'].mean()
            df_rev['Grupo'] = np.where(df_rev['revenue'] > media_receita, 'Head', 'Tail')
            
            df_rev_plot = df_rev
            if len(df_rev_plot) > 3000:
                df_rev_plot = pd.concat([
                    df_rev_plot[df_rev_plot['Grupo'] == 'Head'], 
                    df_rev_plot[df_rev_plot['Grupo'] == 'Tail'].iloc[::max(1, len(df_rev_plot[df_rev_plot['Grupo'] == 'Tail'])//3000)]
                ])

            fig_ht_rev = px.area(df_rev_plot, x=df_rev_plot.index, y='revenue', color='Grupo', color_discrete_map={'Head': '#D90429', 'Tail': '#2B2D42'})
            fig_ht_rev.update_yaxes(type="linear", tickformat=".2s", range=[0, min(800000000, df_rev['revenue'].max())])
            fig_ht_rev.add_hline(y=media_receita, line_dash="dash", line_color="black")
            fig_ht_rev.update_layout(margin=dict(t=20, l=10, r=10, b=10), xaxis_title="Ranking", yaxis_title="Receita (USD)")

            total_filmes = len(df_rev)
            filmes_head = len(df_rev[df_rev['Grupo'] == 'Head'])
            pct_filmes_head = (filmes_head / total_filmes) * 100
            pct_filmes_tail = 100 - pct_filmes_head
            
            receita_head = df_rev[df_rev['Grupo'] == 'Head']['revenue'].sum()
            pct_receita_head = (receita_head / df_rev['revenue'].sum()) * 100
            pct_receita_tail = 100 - pct_receita_head
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='Tail (Cauda)', x=['Volume', 'Receita Total'], y=[pct_filmes_tail, pct_receita_tail], marker_color='#2B2D42', text=[f"{pct_filmes_tail:.1f}%", f"{pct_receita_tail:.1f}%"], textposition='inside'))
            fig_comp.add_trace(go.Bar(name='Head (Cabeça)', x=['Volume', 'Receita Total'], y=[pct_filmes_head, pct_receita_head], marker_color='#D90429', text=[f"{pct_filmes_head:.1f}%", f"{pct_receita_head:.1f}%"], textposition='inside'))
            fig_comp.update_layout(barmode='stack', yaxis=dict(title='Porcentagem (%)', range=[0, 100]), margin=dict(t=20, l=10, r=10, b=10))
        else:
            fig_ht_rev = go.Figure()
            fig_comp = go.Figure()

        df_boxplot = df_filtrado.sort_values(by='vote_count', ascending=False).reset_index(drop=True)
        df_votes_filt = df_boxplot[df_boxplot['vote_count'] > 0].copy()
        
        if not df_votes_filt.empty:
            media_v_filt = df_votes_filt['vote_count'].mean()
            corte_index_filt = len(df_votes_filt[df_votes_filt['vote_count'] > media_v_filt])
            
            notas_disponiveis = df_votes_filt['vote_average'].values
            mask_zero = df_boxplot['vote_count'] == 0
            num_zeros = mask_zero.sum()
            df_boxplot['vote_average_imputed'] = df_boxplot['vote_average']
            
            if num_zeros > 0 and len(notas_disponiveis) > 0:
                notas_imputadas = np.random.choice(notas_disponiveis, size=num_zeros)
                df_boxplot.loc[mask_zero, 'vote_average_imputed'] = notas_imputadas
                
            df_boxplot['Segmento'] = np.where(df_boxplot.index < corte_index_filt, 'Mainstream (Head)', 'Alternativo (Tail)')
            
            fig_qualidade = px.box(
                df_boxplot, x='Segmento', y='vote_average_imputed', color='Segmento',
                color_discrete_map={'Mainstream (Head)': 'royalblue', 'Alternativo (Tail)': 'darkorange'},
                labels={'vote_average_imputed': 'Nota Média', 'Segmento': 'Segmento'}
            )
            fig_qualidade.update_layout(yaxis=dict(range=[0, 10]), margin=dict(t=20, l=10, r=10, b=10), showlegend=False)
        else:
            fig_qualidade = go.Figure()

        media_votos_filtrado = df_filtrado['vote_count'].mean()
        df_geo = df_filtrado[df_filtrado['primeiro_pais'] != 'Desconhecido']
        top_8_countries = df_geo['primeiro_pais'].value_counts().head(8).index.tolist()
        df_geo_filtered = df_geo[df_geo['primeiro_pais'].isin(top_8_countries)].copy()

        if not df_geo_filtered.empty:
            df_geo_filtered['Segmento'] = np.where(df_geo_filtered['vote_count'] >= media_votos_filtrado, 'Cabeça', 'Cauda')
            segment_counts = df_geo_filtered['Segmento'].value_counts()
            country_segment_counts = df_geo_filtered.groupby(['Segmento', 'primeiro_pais']).size().reset_index(name='count')
            
            country_segment_counts['percentage'] = country_segment_counts.apply(
                lambda row: (row['count'] / segment_counts[row['Segmento']]) * 100 if row['Segmento'] in segment_counts else 0, axis=1
            )

            fig_geo = px.bar(
                country_segment_counts, x='primeiro_pais', y='percentage', color='Segmento', barmode='group',
                color_discrete_map={'Cabeça': 'gray', 'Cauda': 'royalblue'},
                text=country_segment_counts['percentage'].apply(lambda x: f'{x:.1f}%')
            )
            fig_geo.update_traces(textposition='outside')
            fig_geo.update_layout(yaxis_title='Proporção (%)', xaxis_title='País', margin=dict(t=20, l=10, r=10, b=10), legend_title_text='')
        else:
            fig_geo = go.Figure()

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

            fig_gen_main = px.bar(df_gen_main, x='Count', y='Genre', orientation='h', color='Count', color_continuous_scale='viridis')
            fig_gen_main.update_layout(title=f"Votos > {int(media_votos_filtrado)}", xaxis_range=[0, max_count], margin=dict(t=30, l=10, r=10, b=10), coloraxis_showscale=False)

            fig_gen_tail = px.bar(df_gen_tail, x='Count', y='Genre', orientation='h', color='Count', color_continuous_scale='viridis')
            fig_gen_tail.update_layout(title=f"Votos <= {int(media_votos_filtrado)}", xaxis_range=[0, max_count], margin=dict(t=30, l=10, r=10, b=10), coloraxis_showscale=False)
        else:
            fig_gen_main = go.Figure()
            fig_gen_tail = go.Figure()

        return fig_ht_rev, fig_comp, fig_qualidade, fig_geo, fig_gen_main, fig_gen_tail