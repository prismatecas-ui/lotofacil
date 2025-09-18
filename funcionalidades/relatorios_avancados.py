#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Relatórios Avançados - Lotofácil

Este módulo implementa um sistema completo de geração de relatórios
avançados com análises estatísticas, gráficos e exportação em múltiplos formatos.

Autor: Sistema Lotofácil
Versão: 1.0.0
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração do matplotlib para português
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

class RelatoriosAvancados:
    """
    Sistema completo de relatórios avançados para análise da Lotofácil.
    
    Funcionalidades:
    - Relatórios estatísticos detalhados
    - Análises de tendências e padrões
    - Gráficos interativos e estáticos
    - Exportação em múltiplos formatos
    - Relatórios personalizáveis
    - Comparações entre períodos
    """
    
    def __init__(self, db_path: str = "database/lotofacil.db"):
        """
        Inicializa o sistema de relatórios.
        
        Args:
            db_path: Caminho para o banco de dados SQLite
        """
        self.db_path = db_path
        self.numeros_lotofacil = list(range(1, 26))
        self.cores_graficos = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Criar diretórios necessários
        Path("funcionalidades/relatorios").mkdir(parents=True, exist_ok=True)
        Path("funcionalidades/templates").mkdir(parents=True, exist_ok=True)
        
    def conectar_db(self) -> sqlite3.Connection:
        """Estabelece conexão com o banco de dados."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            logger.error(f"Erro ao conectar com o banco: {e}")
            raise
    
    def gerar_relatorio_completo(self, 
                                parametros: Dict,
                                formato_saida: str = 'html') -> str:
        """
        Gera relatório completo baseado nos parâmetros fornecidos.
        
        Args:
            parametros: Dicionário com configurações do relatório
            formato_saida: Formato de saída ('html', 'pdf', 'json')
            
        Returns:
            Caminho do arquivo gerado
        """
        try:
            logger.info("Iniciando geração de relatório completo...")
            
            # Validar parâmetros
            parametros = self._validar_parametros(parametros)
            
            # Coletar dados
            dados = self._coletar_dados_relatorio(parametros)
            
            # Gerar análises
            analises = self._gerar_analises_completas(dados, parametros)
            
            # Gerar gráficos
            graficos = self._gerar_graficos_relatorio(dados, parametros)
            
            # Compilar relatório
            relatorio = {
                'metadados': {
                    'titulo': parametros.get('titulo', 'Relatório Lotofácil'),
                    'data_geracao': datetime.now().isoformat(),
                    'periodo_analise': parametros.get('periodo'),
                    'parametros_utilizados': parametros
                },
                'resumo_executivo': self._gerar_resumo_executivo(analises),
                'dados_brutos': dados if parametros.get('incluir_dados_brutos', False) else None,
                'analises': analises,
                'graficos': graficos,
                'conclusoes': self._gerar_conclusoes(analises)
            }
            
            # Exportar no formato solicitado
            arquivo_saida = self._exportar_relatorio(relatorio, formato_saida, parametros)
            
            logger.info(f"Relatório completo gerado: {arquivo_saida}")
            return arquivo_saida
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório completo: {e}")
            raise
    
    def _validar_parametros(self, parametros: Dict) -> Dict:
        """Valida e completa os parâmetros do relatório."""
        # Parâmetros padrão
        defaults = {
            'periodo': {'inicio': '2023-01-01', 'fim': datetime.now().strftime('%Y-%m-%d')},
            'concursos_especificos': None,
            'incluir_graficos': True,
            'incluir_analise_tendencias': True,
            'incluir_comparacoes': True,
            'incluir_previsoes': False,
            'nivel_detalhamento': 'completo',  # basico, intermediario, completo
            'formato_numeros': 'lista',  # lista, matriz
            'incluir_dados_brutos': False
        }
        
        # Mesclar com parâmetros fornecidos
        for key, value in defaults.items():
            if key not in parametros:
                parametros[key] = value
        
        return parametros
    
    def _coletar_dados_relatorio(self, parametros: Dict) -> Dict:
        """Coleta todos os dados necessários para o relatório."""
        try:
            conn = self.conectar_db()
            
            # Determinar quais concursos analisar
            if parametros.get('concursos_especificos'):
                # Concursos específicos
                concursos_lista = parametros['concursos_especificos']
                placeholders = ','.join(['?' for _ in concursos_lista])
                query = f"""
                SELECT * FROM concursos 
                WHERE numero IN ({placeholders})
                ORDER BY numero
                """
                dados_concursos = pd.read_sql_query(query, conn, params=concursos_lista)
            else:
                # Por período
                periodo = parametros['periodo']
                query = """
                SELECT * FROM concursos 
                WHERE data_sorteio BETWEEN ? AND ?
                ORDER BY numero
                """
                dados_concursos = pd.read_sql_query(query, conn, 
                                                   params=(periodo['inicio'], periodo['fim']))
            
            conn.close()
            
            if dados_concursos.empty:
                raise ValueError("Nenhum concurso encontrado para os parâmetros especificados")
            
            # Processar dados dos concursos
            concursos_processados = []
            todos_numeros = []
            
            for _, concurso in dados_concursos.iterrows():
                numeros_sorteados = json.loads(concurso['numeros_sorteados'])
                todos_numeros.extend(numeros_sorteados)
                
                concurso_info = {
                    'concurso': int(concurso['concurso']),
                    'data_sorteio': concurso['data_sorteio'],
                    'numeros_sorteados': numeros_sorteados,
                    'premio_total': concurso.get('premio_total'),
                    'ganhadores': {
                        '15_acertos': concurso.get('ganhadores_15'),
                        '14_acertos': concurso.get('ganhadores_14'),
                        '13_acertos': concurso.get('ganhadores_13'),
                        '12_acertos': concurso.get('ganhadores_12'),
                        '11_acertos': concurso.get('ganhadores_11')
                    }
                }
                
                concursos_processados.append(concurso_info)
            
            return {
                'concursos': concursos_processados,
                'total_concursos': len(concursos_processados),
                'todos_numeros': todos_numeros,
                'periodo_analise': {
                    'primeiro_concurso': concursos_processados[0]['concurso'],
                    'ultimo_concurso': concursos_processados[-1]['concurso'],
                    'primeira_data': concursos_processados[0]['data_sorteio'],
                    'ultima_data': concursos_processados[-1]['data_sorteio']
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao coletar dados: {e}")
            raise
    
    def _gerar_analises_completas(self, dados: Dict, parametros: Dict) -> Dict:
        """Gera todas as análises estatísticas."""
        analises = {}
        
        # Análise de frequência
        analises['frequencia'] = self._analisar_frequencia(dados)
        
        # Análise de padrões
        analises['padroes'] = self._analisar_padroes(dados)
        
        # Análise de tendências (se solicitado)
        if parametros.get('incluir_analise_tendencias', True):
            analises['tendencias'] = self._analisar_tendencias(dados)
        
        # Análise de distribuições
        analises['distribuicoes'] = self._analisar_distribuicoes(dados)
        
        # Análise de correlações
        analises['correlacoes'] = self._analisar_correlacoes(dados)
        
        # Análise de ciclos e sazonalidade
        analises['ciclos'] = self._analisar_ciclos(dados)
        
        # Comparações (se solicitado)
        if parametros.get('incluir_comparacoes', True):
            analises['comparacoes'] = self._gerar_comparacoes(dados)
        
        return analises
    
    def _analisar_frequencia(self, dados: Dict) -> Dict:
        """Análise detalhada de frequência dos números."""
        todos_numeros = dados['todos_numeros']
        total_concursos = dados['total_concursos']
        
        # Frequência absoluta
        frequencia = Counter(todos_numeros)
        
        # Frequência relativa
        frequencia_relativa = {num: freq/total_concursos for num, freq in frequencia.items()}
        
        # Estatísticas
        frequencias = list(frequencia.values())
        media_freq = np.mean(frequencias)
        desvio_freq = np.std(frequencias)
        
        # Números mais e menos sorteados
        mais_sorteados = frequencia.most_common(10)
        menos_sorteados = frequencia.most_common()[:-11:-1]
        
        # Números em falta
        numeros_ausentes = [n for n in self.numeros_lotofacil if n not in frequencia]
        
        # Análise de desvios
        frequencia_esperada = len(todos_numeros) / 25  # Distribuição uniforme esperada
        desvios = {num: freq - frequencia_esperada for num, freq in frequencia.items()}
        
        return {
            'frequencia_absoluta': dict(frequencia),
            'frequencia_relativa': frequencia_relativa,
            'estatisticas': {
                'media': round(media_freq, 2),
                'desvio_padrao': round(desvio_freq, 2),
                'coeficiente_variacao': round(desvio_freq/media_freq, 4),
                'frequencia_esperada': round(frequencia_esperada, 2)
            },
            'rankings': {
                'mais_sorteados': mais_sorteados,
                'menos_sorteados': menos_sorteados
            },
            'numeros_ausentes': numeros_ausentes,
            'desvios_esperado': desvios
        }
    
    def _analisar_padroes(self, dados: Dict) -> Dict:
        """Análise de padrões nos sorteios."""
        concursos = dados['concursos']
        
        # Análise de paridade
        padroes_paridade = []
        sequencias_consecutivas = []
        distribuicao_dezenas = []
        somas_concursos = []
        
        for concurso in concursos:
            numeros = concurso['numeros_sorteados']
            
            # Paridade
            pares = sum(1 for n in numeros if n % 2 == 0)
            padroes_paridade.append(pares)
            
            # Sequências
            numeros_ord = sorted(numeros)
            seq_count = 0
            for i in range(len(numeros_ord)-1):
                if numeros_ord[i+1] == numeros_ord[i] + 1:
                    seq_count += 1
            sequencias_consecutivas.append(seq_count)
            
            # Distribuição por dezenas
            primeira_dez = sum(1 for n in numeros if 1 <= n <= 10)
            segunda_dez = sum(1 for n in numeros if 11 <= n <= 20)
            terceira_dez = sum(1 for n in numeros if 21 <= n <= 25)
            distribuicao_dezenas.append([primeira_dez, segunda_dez, terceira_dez])
            
            # Soma dos números
            somas_concursos.append(sum(numeros))
        
        return {
            'paridade': {
                'distribuicao': Counter(padroes_paridade),
                'media_pares': round(np.mean(padroes_paridade), 2),
                'desvio_pares': round(np.std(padroes_paridade), 2)
            },
            'sequencias': {
                'distribuicao': Counter(sequencias_consecutivas),
                'media_sequencias': round(np.mean(sequencias_consecutivas), 2)
            },
            'distribuicao_dezenas': {
                'media_por_dezena': {
                    'primeira': round(np.mean([d[0] for d in distribuicao_dezenas]), 2),
                    'segunda': round(np.mean([d[1] for d in distribuicao_dezenas]), 2),
                    'terceira': round(np.mean([d[2] for d in distribuicao_dezenas]), 2)
                }
            },
            'somas': {
                'distribuicao': somas_concursos,
                'media': round(np.mean(somas_concursos), 2),
                'desvio': round(np.std(somas_concursos), 2),
                'minima': min(somas_concursos),
                'maxima': max(somas_concursos)
            }
        }
    
    def _analisar_tendencias(self, dados: Dict) -> Dict:
        """Análise de tendências temporais."""
        concursos = dados['concursos']
        
        # Preparar dados temporais
        datas = [datetime.strptime(c['data_sorteio'], '%Y-%m-%d') for c in concursos]
        
        # Tendência de frequência por número ao longo do tempo
        tendencias_numeros = {}
        janela_movel = 50  # Últimos 50 concursos
        
        for numero in self.numeros_lotofacil:
            frequencias_tempo = []
            
            for i in range(janela_movel, len(concursos)):
                concursos_janela = concursos[i-janela_movel:i]
                freq_numero = sum(1 for c in concursos_janela if numero in c['numeros_sorteados'])
                frequencias_tempo.append(freq_numero)
            
            if frequencias_tempo:
                # Calcular tendência (regressão linear simples)
                x = np.arange(len(frequencias_tempo))
                coef = np.polyfit(x, frequencias_tempo, 1)
                tendencias_numeros[numero] = {
                    'coeficiente_angular': round(coef[0], 6),
                    'tendencia': 'crescente' if coef[0] > 0 else 'decrescente' if coef[0] < 0 else 'estável',
                    'frequencias': frequencias_tempo
                }
        
        # Análise de sazonalidade (por mês)
        frequencia_mensal = defaultdict(list)
        for concurso in concursos:
            data = datetime.strptime(concurso['data_sorteio'], '%Y-%m-%d')
            mes = data.month
            for numero in concurso['numeros_sorteados']:
                frequencia_mensal[mes].append(numero)
        
        sazonalidade = {}
        for mes in range(1, 13):
            if mes in frequencia_mensal:
                freq_mes = Counter(frequencia_mensal[mes])
                sazonalidade[mes] = dict(freq_mes)
        
        return {
            'tendencias_numeros': tendencias_numeros,
            'sazonalidade_mensal': sazonalidade,
            'periodo_analise': {
                'data_inicio': min(datas).strftime('%Y-%m-%d'),
                'data_fim': max(datas).strftime('%Y-%m-%d'),
                'total_dias': (max(datas) - min(datas)).days
            }
        }
    
    def _analisar_distribuicoes(self, dados: Dict) -> Dict:
        """Análise de distribuições estatísticas."""
        todos_numeros = dados['todos_numeros']
        concursos = dados['concursos']
        
        # Distribuição na cartela (5x5)
        cartela_freq = np.zeros((5, 5))
        for numero in todos_numeros:
            linha = (numero - 1) // 5
            coluna = (numero - 1) % 5
            cartela_freq[linha, coluna] += 1
        
        # Distribuição por posição no sorteio
        posicoes_freq = defaultdict(list)
        for concurso in concursos:
            numeros_ord = sorted(concurso['numeros_sorteados'])
            for pos, numero in enumerate(numeros_ord):
                posicoes_freq[pos].append(numero)
        
        # Análise de gaps (intervalos entre aparições)
        gaps_numeros = {}
        for numero in self.numeros_lotofacil:
            aparicoes = []
            for i, concurso in enumerate(concursos):
                if numero in concurso['numeros_sorteados']:
                    aparicoes.append(i)
            
            if len(aparicoes) > 1:
                gaps = [aparicoes[i+1] - aparicoes[i] for i in range(len(aparicoes)-1)]
                gaps_numeros[numero] = {
                    'gaps': gaps,
                    'gap_medio': round(np.mean(gaps), 2),
                    'gap_maximo': max(gaps),
                    'gap_minimo': min(gaps)
                }
        
        return {
            'cartela_5x5': cartela_freq.tolist(),
            'posicoes_sorteio': {pos: Counter(nums) for pos, nums in posicoes_freq.items()},
            'gaps_aparicoes': gaps_numeros
        }
    
    def _analisar_correlacoes(self, dados: Dict) -> Dict:
        """Análise de correlações entre números."""
        concursos = dados['concursos']
        
        # Matriz de co-ocorrência
        coocorrencia = np.zeros((25, 25))
        
        for concurso in concursos:
            numeros = concurso['numeros_sorteados']
            for i, num1 in enumerate(numeros):
                for j, num2 in enumerate(numeros):
                    if i != j:
                        coocorrencia[num1-1, num2-1] += 1
        
        # Normalizar para obter correlações
        total_concursos = len(concursos)
        correlacao_normalizada = coocorrencia / total_concursos
        
        # Encontrar pares mais correlacionados
        pares_correlacionados = []
        for i in range(25):
            for j in range(i+1, 25):
                corr = correlacao_normalizada[i, j]
                pares_correlacionados.append(((i+1, j+1), corr))
        
        pares_correlacionados.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'matriz_coocorrencia': coocorrencia.tolist(),
            'correlacao_normalizada': correlacao_normalizada.tolist(),
            'top_pares_correlacionados': pares_correlacionados[:20],
            'pares_menos_correlacionados': pares_correlacionados[-20:]
        }
    
    def _analisar_ciclos(self, dados: Dict) -> Dict:
        """Análise de ciclos e periodicidade."""
        concursos = dados['concursos']
        
        # Análise de ciclos para cada número
        ciclos_numeros = {}
        
        for numero in self.numeros_lotofacil:
            aparicoes = []
            for i, concurso in enumerate(concursos):
                if numero in concurso['numeros_sorteados']:
                    aparicoes.append(i)
            
            if len(aparicoes) >= 3:
                # Calcular intervalos
                intervalos = [aparicoes[i+1] - aparicoes[i] for i in range(len(aparicoes)-1)]
                
                # Detectar possível ciclo
                ciclo_detectado = None
                if len(intervalos) >= 3:
                    # Verificar se há padrão nos intervalos
                    media_intervalo = np.mean(intervalos)
                    desvio_intervalo = np.std(intervalos)
                    
                    if desvio_intervalo < media_intervalo * 0.3:  # Baixa variabilidade
                        ciclo_detectado = round(media_intervalo)
                
                ciclos_numeros[numero] = {
                    'intervalos': intervalos,
                    'media_intervalo': round(np.mean(intervalos), 2),
                    'desvio_intervalo': round(np.std(intervalos), 2),
                    'ciclo_detectado': ciclo_detectado,
                    'ultima_aparicao': max(aparicoes),
                    'proxima_previsao': max(aparicoes) + ciclo_detectado if ciclo_detectado else None
                }
        
        return {
            'ciclos_por_numero': ciclos_numeros,
            'numeros_com_ciclo': {k: v for k, v in ciclos_numeros.items() if v['ciclo_detectado']}
        }
    
    def _gerar_comparacoes(self, dados: Dict) -> Dict:
        """Gera comparações entre diferentes períodos."""
        concursos = dados['concursos']
        total = len(concursos)
        
        if total < 20:  # Mínimo para comparação
            return {'erro': 'Dados insuficientes para comparação'}
        
        # Dividir em dois períodos
        meio = total // 2
        primeiro_periodo = concursos[:meio]
        segundo_periodo = concursos[meio:]
        
        # Análise comparativa de frequência
        freq_primeiro = Counter()
        freq_segundo = Counter()
        
        for concurso in primeiro_periodo:
            freq_primeiro.update(concurso['numeros_sorteados'])
        
        for concurso in segundo_periodo:
            freq_segundo.update(concurso['numeros_sorteados'])
        
        # Calcular diferenças
        diferencas = {}
        for numero in self.numeros_lotofacil:
            freq1 = freq_primeiro.get(numero, 0) / len(primeiro_periodo)
            freq2 = freq_segundo.get(numero, 0) / len(segundo_periodo)
            diferencas[numero] = freq2 - freq1
        
        # Números com maior mudança
        maiores_aumentos = sorted(diferencas.items(), key=lambda x: x[1], reverse=True)[:10]
        maiores_diminuicoes = sorted(diferencas.items(), key=lambda x: x[1])[:10]
        
        return {
            'periodos': {
                'primeiro': {
                    'concursos': len(primeiro_periodo),
                    'periodo': f"{primeiro_periodo[0]['data_sorteio']} a {primeiro_periodo[-1]['data_sorteio']}"
                },
                'segundo': {
                    'concursos': len(segundo_periodo),
                    'periodo': f"{segundo_periodo[0]['data_sorteio']} a {segundo_periodo[-1]['data_sorteio']}"
                }
            },
            'frequencias_comparadas': {
                'primeiro_periodo': dict(freq_primeiro),
                'segundo_periodo': dict(freq_segundo)
            },
            'diferencas': diferencas,
            'maiores_mudancas': {
                'aumentos': maiores_aumentos,
                'diminuicoes': maiores_diminuicoes
            }
        }
    
    def _gerar_graficos_relatorio(self, dados: Dict, parametros: Dict) -> Dict:
        """Gera todos os gráficos para o relatório."""
        if not parametros.get('incluir_graficos', True):
            return {}
        
        graficos = {}
        
        try:
            # Gráfico de frequência
            graficos['frequencia'] = self._criar_grafico_frequencia(dados)
            
            # Gráfico de tendências
            graficos['tendencias'] = self._criar_grafico_tendencias(dados)
            
            # Heatmap da cartela
            graficos['heatmap_cartela'] = self._criar_heatmap_cartela(dados)
            
            # Gráfico de distribuição de somas
            graficos['distribuicao_somas'] = self._criar_grafico_somas(dados)
            
            # Gráfico de paridade
            graficos['paridade'] = self._criar_grafico_paridade(dados)
            
            return graficos
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráficos: {e}")
            return {'erro': str(e)}
    
    def _criar_grafico_frequencia(self, dados: Dict) -> str:
        """Cria gráfico de frequência dos números."""
        todos_numeros = dados['todos_numeros']
        frequencia = Counter(todos_numeros)
        
        # Preparar dados
        numeros = sorted(frequencia.keys())
        freqs = [frequencia[n] for n in numeros]
        
        # Criar gráfico
        fig = go.Figure(data=[
            go.Bar(x=numeros, y=freqs, 
                  marker_color='lightblue',
                  text=freqs,
                  textposition='auto')
        ])
        
        fig.update_layout(
            title='Frequência dos Números',
            xaxis_title='Números',
            yaxis_title='Frequência',
            showlegend=False
        )
        
        # Salvar como HTML
        arquivo = f"funcionalidades/relatorios/grafico_frequencia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(arquivo)
        
        return arquivo
    
    def _criar_grafico_tendencias(self, dados: Dict) -> str:
        """Cria gráfico de tendências temporais."""
        concursos = dados['concursos']
        
        # Preparar dados para alguns números principais
        numeros_principais = [1, 5, 10, 15, 20, 25]
        
        fig = go.Figure()
        
        for numero in numeros_principais:
            frequencias = []
            concursos_x = []
            
            janela = 20
            for i in range(janela, len(concursos)):
                concursos_janela = concursos[i-janela:i]
                freq = sum(1 for c in concursos_janela if numero in c['numeros_sorteados'])
                frequencias.append(freq)
                concursos_x.append(concursos[i]['concurso'])
            
            fig.add_trace(go.Scatter(
                x=concursos_x,
                y=frequencias,
                mode='lines',
                name=f'Número {numero}'
            ))
        
        fig.update_layout(
            title='Tendências de Frequência (Janela Móvel de 20 concursos)',
            xaxis_title='Concurso',
            yaxis_title='Frequência na Janela',
            hovermode='x unified'
        )
        
        arquivo = f"funcionalidades/relatorios/grafico_tendencias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(arquivo)
        
        return arquivo
    
    def _criar_heatmap_cartela(self, dados: Dict) -> str:
        """Cria heatmap da cartela 5x5."""
        todos_numeros = dados['todos_numeros']
        
        # Criar matriz 5x5
        cartela = np.zeros((5, 5))
        for numero in todos_numeros:
            linha = (numero - 1) // 5
            coluna = (numero - 1) % 5
            cartela[linha, coluna] += 1
        
        # Criar labels da cartela
        labels = np.zeros((5, 5), dtype=int)
        for i in range(5):
            for j in range(5):
                labels[i, j] = i * 5 + j + 1
        
        fig = go.Figure(data=go.Heatmap(
            z=cartela,
            text=labels,
            texttemplate="%{text}<br>%{z}",
            textfont={"size": 12},
            colorscale='YlOrRd'
        ))
        
        fig.update_layout(
            title='Mapa de Calor da Cartela Lotofácil',
            xaxis_title='Colunas',
            yaxis_title='Linhas'
        )
        
        arquivo = f"funcionalidades/relatorios/heatmap_cartela_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(arquivo)
        
        return arquivo
    
    def _criar_grafico_somas(self, dados: Dict) -> str:
        """Cria gráfico de distribuição das somas."""
        concursos = dados['concursos']
        somas = [sum(c['numeros_sorteados']) for c in concursos]
        
        fig = go.Figure(data=[go.Histogram(x=somas, nbinsx=20, marker_color='lightgreen')])
        
        fig.update_layout(
            title='Distribuição das Somas dos Números Sorteados',
            xaxis_title='Soma',
            yaxis_title='Frequência'
        )
        
        arquivo = f"funcionalidades/relatorios/grafico_somas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(arquivo)
        
        return arquivo
    
    def _criar_grafico_paridade(self, dados: Dict) -> str:
        """Cria gráfico de distribuição de paridade."""
        concursos = dados['concursos']
        pares_por_concurso = []
        
        for concurso in concursos:
            pares = sum(1 for n in concurso['numeros_sorteados'] if n % 2 == 0)
            pares_por_concurso.append(pares)
        
        distribuicao = Counter(pares_por_concurso)
        
        fig = go.Figure(data=[
            go.Bar(x=list(distribuicao.keys()), 
                  y=list(distribuicao.values()),
                  marker_color='lightcoral')
        ])
        
        fig.update_layout(
            title='Distribuição de Números Pares por Concurso',
            xaxis_title='Quantidade de Números Pares',
            yaxis_title='Frequência'
        )
        
        arquivo = f"funcionalidades/relatorios/grafico_paridade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(arquivo)
        
        return arquivo
    
    def _gerar_resumo_executivo(self, analises: Dict) -> Dict:
        """Gera resumo executivo das análises."""
        resumo = {
            'principais_insights': [],
            'numeros_destaque': {},
            'padroes_identificados': [],
            'recomendacoes': []
        }
        
        # Análise de frequência
        freq_data = analises['frequencia']
        mais_sorteados = freq_data['rankings']['mais_sorteados'][:5]
        menos_sorteados = freq_data['rankings']['menos_sorteados'][:5]
        
        resumo['numeros_destaque'] = {
            'mais_frequentes': [num for num, freq in mais_sorteados],
            'menos_frequentes': [num for num, freq in menos_sorteados]
        }
        
        # Insights principais
        resumo['principais_insights'].append(
            f"Os números mais sorteados são: {', '.join(map(str, resumo['numeros_destaque']['mais_frequentes']))}"
        )
        
        resumo['principais_insights'].append(
            f"Os números menos sorteados são: {', '.join(map(str, resumo['numeros_destaque']['menos_frequentes']))}"
        )
        
        # Padrões de paridade
        if 'padroes' in analises:
            media_pares = analises['padroes']['paridade']['media_pares']
            resumo['padroes_identificados'].append(
                f"Média de números pares por concurso: {media_pares}"
            )
        
        # Recomendações
        resumo['recomendacoes'] = [
            "Considere a distribuição histórica de frequências na escolha dos números",
            "Observe os padrões de paridade para balancear a aposta",
            "Analise as tendências temporais para identificar números em alta ou baixa"
        ]
        
        return resumo
    
    def _gerar_conclusoes(self, analises: Dict) -> Dict:
        """Gera conclusões baseadas nas análises."""
        conclusoes = {
            'estatisticas_gerais': {},
            'observacoes_importantes': [],
            'limitacoes_analise': [],
            'sugestoes_futuras': []
        }
        
        # Estatísticas gerais
        if 'frequencia' in analises:
            freq_stats = analises['frequencia']['estatisticas']
            conclusoes['estatisticas_gerais'] = {
                'coeficiente_variacao': freq_stats['coeficiente_variacao'],
                'distribuicao': 'uniforme' if freq_stats['coeficiente_variacao'] < 0.1 else 'não-uniforme'
            }
        
        # Observações importantes
        conclusoes['observacoes_importantes'] = [
            "A Lotofácil apresenta características de aleatoriedade",
            "Padrões históricos não garantem resultados futuros",
            "A análise estatística pode auxiliar na compreensão dos dados"
        ]
        
        # Limitações
        conclusoes['limitacoes_analise'] = [
            "Análise baseada em dados históricos",
            "Resultados passados não influenciam sorteios futuros",
            "Aleatoriedade é característica fundamental dos sorteios"
        ]
        
        # Sugestões futuras
        conclusoes['sugestoes_futuras'] = [
            "Implementar análises preditivas com machine learning",
            "Desenvolver modelos de simulação Monte Carlo",
            "Criar alertas para padrões anômalos"
        ]
        
        return conclusoes
    
    def _exportar_relatorio(self, relatorio: Dict, formato: str, parametros: Dict) -> str:
        """Exporta o relatório no formato especificado."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if formato.lower() == 'json':
            arquivo = f"funcionalidades/relatorios/relatorio_completo_{timestamp}.json"
            with open(arquivo, 'w', encoding='utf-8') as f:
                json.dump(relatorio, f, indent=2, ensure_ascii=False, default=str)
        
        elif formato.lower() == 'html':
            arquivo = f"funcionalidades/relatorios/relatorio_completo_{timestamp}.html"
            html_content = self._gerar_html_relatorio(relatorio)
            with open(arquivo, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Formato '{formato}' não suportado")
        
        return arquivo
    
    def _gerar_html_relatorio(self, relatorio: Dict) -> str:
        """Gera HTML do relatório."""
        template_html = """
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ titulo }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }
                .insight { background-color: #e7f3ff; padding: 10px; margin: 10px 0; border-radius: 3px; }
                .number-list { display: inline-block; margin: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ titulo }}</h1>
                <p><strong>Data de Geração:</strong> {{ data_geracao }}</p>
                <p><strong>Período de Análise:</strong> {{ periodo }}</p>
            </div>
            
            <div class="section">
                <h2>Resumo Executivo</h2>
                {% for insight in resumo.principais_insights %}
                <div class="insight">{{ insight }}</div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>Números em Destaque</h2>
                <p><strong>Mais Frequentes:</strong> 
                {% for num in resumo.numeros_destaque.mais_frequentes %}
                <span class="number-list">{{ num }}</span>
                {% endfor %}
                </p>
                <p><strong>Menos Frequentes:</strong> 
                {% for num in resumo.numeros_destaque.menos_frequentes %}
                <span class="number-list">{{ num }}</span>
                {% endfor %}
                </p>
            </div>
            
            <div class="section">
                <h2>Conclusões</h2>
                {% for obs in conclusoes.observacoes_importantes %}
                <p>• {{ obs }}</p>
                {% endfor %}
            </div>
        </body>
        </html>
        """
        
        template = Template(template_html)
        
        return template.render(
            titulo=relatorio['metadados']['titulo'],
            data_geracao=relatorio['metadados']['data_geracao'],
            periodo=str(relatorio['metadados'].get('periodo_analise', 'N/A')),
            resumo=relatorio['resumo_executivo'],
            conclusoes=relatorio['conclusoes']
        )


def exemplo_uso():
    """
    Exemplo de uso do sistema de relatórios avançados.
    """
    # Inicializar sistema
    relatorios = RelatoriosAvancados()
    
    try:
        # Parâmetros do relatório
        parametros = {
            'titulo': 'Análise Completa Lotofácil - Últimos 100 Concursos',
            'periodo': {
                'inicio': '2023-01-01',
                'fim': '2024-01-31'
            },
            'incluir_graficos': True,
            'incluir_analise_tendencias': True,
            'incluir_comparacoes': True,
            'nivel_detalhamento': 'completo'
        }
        
        # Gerar relatório completo
        print("Gerando relatório completo...")
        arquivo_relatorio = relatorios.gerar_relatorio_completo(parametros, 'html')
        print(f"Relatório gerado: {arquivo_relatorio}")
        
        # Também gerar em JSON para análise programática
        arquivo_json = relatorios.gerar_relatorio_completo(parametros, 'json')
        print(f"Dados em JSON: {arquivo_json}")
        
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    exemplo_uso()