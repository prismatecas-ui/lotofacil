#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard de Estatísticas em Tempo Real - Lotofácil

Este módulo implementa um dashboard web interativo para visualização
de estatísticas da Lotofácil em tempo real com atualizações automáticas.

Autor: Sistema Lotofácil
Versão: 1.0.0
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import asyncio
import websockets
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from collections import Counter, defaultdict
import threading
import time
from functools import wraps
import requests
from concurrent.futures import ThreadPoolExecutor
import schedule

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardTempoReal:
    """
    Dashboard interativo para estatísticas da Lotofácil em tempo real.
    
    Funcionalidades:
    - Interface web responsiva
    - Gráficos interativos com Plotly
    - Atualizações automáticas via WebSocket
    - Métricas em tempo real
    - Alertas e notificações
    - API REST para dados
    - Cache inteligente
    """
    
    def __init__(self, db_path: str = "database/lotofacil.db", port: int = 5000):
        """
        Inicializa o dashboard.
        
        Args:
            db_path: Caminho para o banco de dados
            port: Porta do servidor web
        """
        self.db_path = db_path
        self.port = port
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'lotofacil_dashboard_2024'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Cache para otimização
        self.cache = {
            'dados_basicos': None,
            'ultima_atualizacao': None,
            'tempo_cache': 300  # 5 minutos
        }
        
        # Configurar rotas
        self._configurar_rotas()
        self._configurar_websockets()
        
        # Criar diretórios necessários
        Path("funcionalidades/templates").mkdir(parents=True, exist_ok=True)
        Path("funcionalidades/static/css").mkdir(parents=True, exist_ok=True)
        Path("funcionalidades/static/js").mkdir(parents=True, exist_ok=True)
        
        # Gerar arquivos estáticos
        self._gerar_templates()
        self._gerar_arquivos_estaticos()
        
        # Executor para tarefas assíncronas
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Flag para controle de atualizações
        self.atualizacoes_ativas = False
    
    def conectar_db(self) -> sqlite3.Connection:
        """Estabelece conexão com o banco de dados."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Erro ao conectar com o banco: {e}")
            raise
    
    def _configurar_rotas(self):
        """Configura as rotas da aplicação Flask."""
        
        @self.app.route('/')
        def index():
            """Página principal do dashboard."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/dados-basicos')
        def api_dados_basicos():
            """API para dados básicos do dashboard."""
            try:
                dados = self._obter_dados_basicos()
                return jsonify(dados)
            except Exception as e:
                logger.error(f"Erro na API dados básicos: {e}")
                return jsonify({'erro': str(e)}), 500
        
        @self.app.route('/api/frequencia-numeros')
        def api_frequencia_numeros():
            """API para frequência dos números."""
            try:
                limite = request.args.get('limite', 100, type=int)
                dados = self._obter_frequencia_numeros(limite)
                return jsonify(dados)
            except Exception as e:
                logger.error(f"Erro na API frequência: {e}")
                return jsonify({'erro': str(e)}), 500
        
        @self.app.route('/api/tendencias')
        def api_tendencias():
            """API para análise de tendências."""
            try:
                janela = request.args.get('janela', 20, type=int)
                numeros = request.args.getlist('numeros', type=int)
                if not numeros:
                    numeros = [1, 5, 10, 15, 20, 25]  # Números padrão
                
                dados = self._obter_tendencias(janela, numeros)
                return jsonify(dados)
            except Exception as e:
                logger.error(f"Erro na API tendências: {e}")
                return jsonify({'erro': str(e)}), 500
        
        @self.app.route('/api/estatisticas-tempo-real')
        def api_estatisticas_tempo_real():
            """API para estatísticas em tempo real."""
            try:
                dados = self._obter_estatisticas_tempo_real()
                return jsonify(dados)
            except Exception as e:
                logger.error(f"Erro na API tempo real: {e}")
                return jsonify({'erro': str(e)}), 500
        
        @self.app.route('/api/grafico/<tipo>')
        def api_grafico(tipo):
            """API para gerar gráficos específicos."""
            try:
                parametros = request.args.to_dict()
                grafico = self._gerar_grafico_api(tipo, parametros)
                return jsonify(grafico)
            except Exception as e:
                logger.error(f"Erro na API gráfico {tipo}: {e}")
                return jsonify({'erro': str(e)}), 500
        
        @self.app.route('/api/alertas')
        def api_alertas():
            """API para obter alertas ativos."""
            try:
                alertas = self._verificar_alertas()
                return jsonify(alertas)
            except Exception as e:
                logger.error(f"Erro na API alertas: {e}")
                return jsonify({'erro': str(e)}), 500
    
    def _configurar_websockets(self):
        """Configura os eventos WebSocket."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Cliente conectado."""
            logger.info("Cliente conectado ao WebSocket")
            emit('status', {'mensagem': 'Conectado ao dashboard'})
            
            # Enviar dados iniciais
            try:
                dados_iniciais = self._obter_dados_basicos()
                emit('dados_iniciais', dados_iniciais)
            except Exception as e:
                logger.error(f"Erro ao enviar dados iniciais: {e}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Cliente desconectado."""
            logger.info("Cliente desconectado do WebSocket")
        
        @self.socketio.on('solicitar_atualizacao')
        def handle_solicitar_atualizacao(data):
            """Cliente solicita atualização de dados."""
            try:
                tipo = data.get('tipo', 'geral')
                dados = self._obter_dados_atualizacao(tipo)
                emit('atualizacao_dados', {'tipo': tipo, 'dados': dados})
            except Exception as e:
                logger.error(f"Erro ao processar solicitação: {e}")
                emit('erro', {'mensagem': str(e)})
        
        @self.socketio.on('configurar_alertas')
        def handle_configurar_alertas(data):
            """Configura alertas personalizados."""
            try:
                self._configurar_alertas_usuario(data)
                emit('alertas_configurados', {'status': 'sucesso'})
            except Exception as e:
                logger.error(f"Erro ao configurar alertas: {e}")
                emit('erro', {'mensagem': str(e)})
    
    def _obter_dados_basicos(self) -> Dict:
        """Obtém dados básicos para o dashboard."""
        # Verificar cache
        agora = datetime.now()
        if (self.cache['dados_basicos'] and 
            self.cache['ultima_atualizacao'] and
            (agora - self.cache['ultima_atualizacao']).seconds < self.cache['tempo_cache']):
            return self.cache['dados_basicos']
        
        try:
            conn = self.conectar_db()
            
            # Informações gerais
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM concursos")
            total_concursos = cursor.fetchone()['total']
            
            cursor.execute("""
                SELECT concurso, data_sorteio, numeros_sorteados 
                FROM concursos 
                ORDER BY concurso DESC 
                LIMIT 1
            """)
            ultimo_concurso = cursor.fetchone()
            
            # Frequência geral
            cursor.execute("""
                SELECT numeros_sorteados FROM concursos 
                ORDER BY concurso DESC 
                LIMIT 100
            """)
            resultados_recentes = cursor.fetchall()
            
            todos_numeros = []
            for resultado in resultados_recentes:
                numeros = json.loads(resultado['numeros_sorteados'])
                todos_numeros.extend(numeros)
            
            frequencia = Counter(todos_numeros)
            
            # Estatísticas rápidas
            numeros_quentes = frequencia.most_common(5)
            numeros_frios = frequencia.most_common()[:-6:-1]
            
            conn.close()
            
            dados = {
                'timestamp': agora.isoformat(),
                'total_concursos': total_concursos,
                'ultimo_concurso': {
                    'numero': ultimo_concurso['concurso'] if ultimo_concurso else 0,
                    'data': ultimo_concurso['data_sorteio'] if ultimo_concurso else '',
                    'numeros': json.loads(ultimo_concurso['numeros_sorteados']) if ultimo_concurso else []
                },
                'estatisticas_rapidas': {
                    'numeros_quentes': numeros_quentes,
                    'numeros_frios': numeros_frios,
                    'total_sorteios_analisados': len(todos_numeros)
                },
                'status_sistema': {
                    'online': True,
                    'ultima_atualizacao': agora.strftime('%H:%M:%S'),
                    'atualizacoes_ativas': self.atualizacoes_ativas
                }
            }
            
            # Atualizar cache
            self.cache['dados_basicos'] = dados
            self.cache['ultima_atualizacao'] = agora
            
            return dados
            
        except Exception as e:
            logger.error(f"Erro ao obter dados básicos: {e}")
            raise
    
    def _obter_frequencia_numeros(self, limite: int = 100) -> Dict:
        """Obtém frequência detalhada dos números."""
        try:
            conn = self.conectar_db()
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT numeros_sorteados FROM concursos 
                ORDER BY concurso DESC 
                LIMIT ?
            """, (limite,))
            
            resultados = cursor.fetchall()
            conn.close()
            
            # Processar frequências
            todos_numeros = []
            for resultado in resultados:
                numeros = json.loads(resultado['numeros_sorteados'])
                todos_numeros.extend(numeros)
            
            frequencia = Counter(todos_numeros)
            
            # Preparar dados para gráfico
            numeros_ordenados = sorted(range(1, 26), key=lambda x: frequencia.get(x, 0), reverse=True)
            frequencias_ordenadas = [frequencia.get(n, 0) for n in numeros_ordenados]
            
            # Calcular estatísticas
            media_freq = np.mean(list(frequencia.values()))
            desvio_freq = np.std(list(frequencia.values()))
            
            return {
                'numeros': numeros_ordenados,
                'frequencias': frequencias_ordenadas,
                'frequencia_completa': dict(frequencia),
                'estatisticas': {
                    'media': round(media_freq, 2),
                    'desvio_padrao': round(desvio_freq, 2),
                    'total_sorteios': len(todos_numeros),
                    'concursos_analisados': limite
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter frequência: {e}")
            raise
    
    def _obter_tendencias(self, janela: int, numeros: List[int]) -> Dict:
        """Obtém dados de tendências para números específicos."""
        try:
            conn = self.conectar_db()
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT numero, numeros_sorteados FROM concursos 
                ORDER BY numero ASC
            """)
            
            todos_concursos = cursor.fetchall()
            conn.close()
            
            if len(todos_concursos) < janela:
                raise ValueError(f"Dados insuficientes. Mínimo: {janela} concursos")
            
            # Calcular tendências
            tendencias = {}
            
            for numero in numeros:
                frequencias_janela = []
                concursos_x = []
                
                for i in range(janela, len(todos_concursos)):
                    # Analisar janela móvel
                    concursos_janela = todos_concursos[i-janela:i]
                    freq_numero = 0
                    
                    for concurso in concursos_janela:
                        numeros_sorteados = json.loads(concurso['numeros_sorteados'])
                        if numero in numeros_sorteados:
                            freq_numero += 1
                    
                    frequencias_janela.append(freq_numero)
                    concursos_x.append(todos_concursos[i]['concurso'])
                
                # Calcular tendência (regressão linear)
                if len(frequencias_janela) > 1:
                    x = np.arange(len(frequencias_janela))
                    coef = np.polyfit(x, frequencias_janela, 1)
                    
                    tendencias[numero] = {
                        'concursos': concursos_x,
                        'frequencias': frequencias_janela,
                        'coeficiente_angular': float(coef[0]),
                        'tendencia': 'crescente' if coef[0] > 0.01 else 'decrescente' if coef[0] < -0.01 else 'estável',
                        'ultima_frequencia': frequencias_janela[-1],
                        'media_frequencia': round(np.mean(frequencias_janela), 2)
                    }
            
            return {
                'tendencias': tendencias,
                'parametros': {
                    'janela': janela,
                    'numeros_analisados': numeros,
                    'total_pontos': len(frequencias_janela) if frequencias_janela else 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter tendências: {e}")
            raise
    
    def _obter_estatisticas_tempo_real(self) -> Dict:
        """Obtém estatísticas em tempo real."""
        try:
            conn = self.conectar_db()
            
            # Últimos 10 concursos
            cursor = conn.cursor()
            cursor.execute("""
                SELECT concurso, data_sorteio, numeros_sorteados 
                FROM concursos 
                ORDER BY concurso DESC 
                LIMIT 10
            """)
            
            ultimos_concursos = cursor.fetchall()
            
            # Análise rápida
            todos_numeros_recentes = []
            padroes_paridade = []
            somas_concursos = []
            
            for concurso in ultimos_concursos:
                numeros = json.loads(concurso['numeros_sorteados'])
                todos_numeros_recentes.extend(numeros)
                
                # Paridade
                pares = sum(1 for n in numeros if n % 2 == 0)
                padroes_paridade.append(pares)
                
                # Soma
                somas_concursos.append(sum(numeros))
            
            # Frequência recente
            freq_recente = Counter(todos_numeros_recentes)
            
            # Comparar com média histórica
            cursor.execute("""
                SELECT numeros_sorteados FROM concursos 
                ORDER BY concurso DESC 
                LIMIT 100
            """)
            
            historico = cursor.fetchall()
            todos_numeros_historico = []
            for h in historico:
                todos_numeros_historico.extend(json.loads(h['numeros_sorteados']))
            
            freq_historica = Counter(todos_numeros_historico)
            media_historica = len(todos_numeros_historico) / 25
            
            conn.close()
            
            # Detectar anomalias
            anomalias = []
            for numero in range(1, 26):
                freq_rec = freq_recente.get(numero, 0)
                freq_hist = freq_historica.get(numero, 0) / len(historico) * 10  # Normalizar para 10 concursos
                
                if abs(freq_rec - freq_hist) > 2:  # Threshold para anomalia
                    anomalias.append({
                        'numero': numero,
                        'frequencia_recente': freq_rec,
                        'frequencia_esperada': round(freq_hist, 1),
                        'tipo': 'alta' if freq_rec > freq_hist else 'baixa'
                    })
            
            return {
                'ultimos_concursos': [
                    {
                        'concurso': c['concurso'],
                        'data': c['data_sorteio'],
                        'numeros': json.loads(c['numeros_sorteados'])
                    } for c in ultimos_concursos
                ],
                'frequencia_recente': dict(freq_recente),
                'estatisticas_rapidas': {
                    'media_pares': round(np.mean(padroes_paridade), 1),
                    'media_soma': round(np.mean(somas_concursos), 1),
                    'soma_minima': min(somas_concursos),
                    'soma_maxima': max(somas_concursos)
                },
                'anomalias': anomalias,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas tempo real: {e}")
            raise
    
    def _gerar_grafico_api(self, tipo: str, parametros: Dict) -> Dict:
        """Gera gráficos via API."""
        try:
            if tipo == 'frequencia':
                return self._grafico_frequencia_api(parametros)
            elif tipo == 'tendencias':
                return self._grafico_tendencias_api(parametros)
            elif tipo == 'heatmap':
                return self._grafico_heatmap_api(parametros)
            elif tipo == 'distribuicao':
                return self._grafico_distribuicao_api(parametros)
            else:
                raise ValueError(f"Tipo de gráfico '{tipo}' não suportado")
                
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico {tipo}: {e}")
            raise
    
    def _grafico_frequencia_api(self, parametros: Dict) -> Dict:
        """Gera gráfico de frequência via API."""
        limite = int(parametros.get('limite', 100))
        dados = self._obter_frequencia_numeros(limite)
        
        fig = go.Figure(data=[
            go.Bar(
                x=dados['numeros'],
                y=dados['frequencias'],
                marker_color='lightblue',
                text=dados['frequencias'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f'Frequência dos Números (Últimos {limite} concursos)',
            xaxis_title='Números',
            yaxis_title='Frequência',
            showlegend=False,
            height=400
        )
        
        return {
            'grafico': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
            'dados': dados,
            'tipo': 'frequencia'
        }
    
    def _grafico_tendencias_api(self, parametros: Dict) -> Dict:
        """Gera gráfico de tendências via API."""
        janela = int(parametros.get('janela', 20))
        numeros_str = parametros.get('numeros', '1,5,10,15,20,25')
        numeros = [int(n.strip()) for n in numeros_str.split(',')]
        
        dados = self._obter_tendencias(janela, numeros)
        
        fig = go.Figure()
        
        cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, numero in enumerate(numeros):
            if numero in dados['tendencias']:
                tend = dados['tendencias'][numero]
                fig.add_trace(go.Scatter(
                    x=tend['concursos'],
                    y=tend['frequencias'],
                    mode='lines+markers',
                    name=f'Número {numero}',
                    line=dict(color=cores[i % len(cores)])
                ))
        
        fig.update_layout(
            title=f'Tendências de Frequência (Janela móvel: {janela} concursos)',
            xaxis_title='Concurso',
            yaxis_title='Frequência na Janela',
            hovermode='x unified',
            height=400
        )
        
        return {
            'grafico': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
            'dados': dados,
            'tipo': 'tendencias'
        }
    
    def _grafico_heatmap_api(self, parametros: Dict) -> Dict:
        """Gera heatmap da cartela via API."""
        limite = int(parametros.get('limite', 100))
        dados_freq = self._obter_frequencia_numeros(limite)
        
        # Criar matriz 5x5
        cartela = np.zeros((5, 5))
        labels = np.zeros((5, 5), dtype=int)
        
        for numero in range(1, 26):
            linha = (numero - 1) // 5
            coluna = (numero - 1) % 5
            cartela[linha, coluna] = dados_freq['frequencia_completa'].get(numero, 0)
            labels[linha, coluna] = numero
        
        fig = go.Figure(data=go.Heatmap(
            z=cartela,
            text=labels,
            texttemplate="%{text}<br>%{z}",
            textfont={"size": 12},
            colorscale='YlOrRd',
            showscale=True
        ))
        
        fig.update_layout(
            title=f'Mapa de Calor da Cartela (Últimos {limite} concursos)',
            xaxis_title='Colunas',
            yaxis_title='Linhas',
            height=400
        )
        
        return {
            'grafico': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
            'dados': {'cartela': cartela.tolist(), 'labels': labels.tolist()},
            'tipo': 'heatmap'
        }
    
    def _grafico_distribuicao_api(self, parametros: Dict) -> Dict:
        """Gera gráfico de distribuição via API."""
        dados = self._obter_estatisticas_tempo_real()
        
        # Distribuição de paridade dos últimos concursos
        pares_por_concurso = []
        for concurso in dados['ultimos_concursos']:
            pares = sum(1 for n in concurso['numeros'] if n % 2 == 0)
            pares_por_concurso.append(pares)
        
        distribuicao = Counter(pares_por_concurso)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(distribuicao.keys()),
                y=list(distribuicao.values()),
                marker_color='lightcoral',
                text=list(distribuicao.values()),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Distribuição de Números Pares (Últimos 10 concursos)',
            xaxis_title='Quantidade de Números Pares',
            yaxis_title='Frequência',
            showlegend=False,
            height=400
        )
        
        return {
            'grafico': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
            'dados': {'distribuicao': dict(distribuicao)},
            'tipo': 'distribuicao'
        }
    
    def _verificar_alertas(self) -> Dict:
        """Verifica e retorna alertas ativos."""
        try:
            alertas = []
            
            # Obter dados recentes
            dados_tempo_real = self._obter_estatisticas_tempo_real()
            
            # Verificar anomalias
            for anomalia in dados_tempo_real['anomalias']:
                if anomalia['tipo'] == 'alta':
                    alertas.append({
                        'tipo': 'numero_quente',
                        'prioridade': 'media',
                        'mensagem': f"Número {anomalia['numero']} com frequência alta recente: {anomalia['frequencia_recente']} (esperado: {anomalia['frequencia_esperada']})",
                        'timestamp': datetime.now().isoformat(),
                        'dados': anomalia
                    })
                elif anomalia['tipo'] == 'baixa':
                    alertas.append({
                        'tipo': 'numero_frio',
                        'prioridade': 'baixa',
                        'mensagem': f"Número {anomalia['numero']} com frequência baixa recente: {anomalia['frequencia_recente']} (esperado: {anomalia['frequencia_esperada']})",
                        'timestamp': datetime.now().isoformat(),
                        'dados': anomalia
                    })
            
            # Verificar padrões extremos
            stats = dados_tempo_real['estatisticas_rapidas']
            if stats['media_pares'] > 9 or stats['media_pares'] < 5:
                alertas.append({
                    'tipo': 'paridade_extrema',
                    'prioridade': 'alta',
                    'mensagem': f"Padrão de paridade extremo detectado: média de {stats['media_pares']} números pares",
                    'timestamp': datetime.now().isoformat(),
                    'dados': {'media_pares': stats['media_pares']}
                })
            
            return {
                'alertas': alertas,
                'total': len(alertas),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao verificar alertas: {e}")
            return {'alertas': [], 'total': 0, 'erro': str(e)}
    
    def _obter_dados_atualizacao(self, tipo: str) -> Dict:
        """Obtém dados para atualização via WebSocket."""
        if tipo == 'geral':
            return self._obter_dados_basicos()
        elif tipo == 'frequencia':
            return self._obter_frequencia_numeros()
        elif tipo == 'tempo_real':
            return self._obter_estatisticas_tempo_real()
        elif tipo == 'alertas':
            return self._verificar_alertas()
        else:
            raise ValueError(f"Tipo de atualização '{tipo}' não suportado")
    
    def _configurar_alertas_usuario(self, configuracao: Dict):
        """Configura alertas personalizados do usuário."""
        # Implementar lógica de configuração de alertas
        # Por enquanto, apenas log
        logger.info(f"Configuração de alertas recebida: {configuracao}")
    
    def iniciar_atualizacoes_automaticas(self):
        """Inicia o sistema de atualizações automáticas."""
        self.atualizacoes_ativas = True
        
        def worker_atualizacoes():
            while self.atualizacoes_ativas:
                try:
                    # Invalidar cache
                    self.cache['dados_basicos'] = None
                    
                    # Obter dados atualizados
                    dados = self._obter_dados_basicos()
                    alertas = self._verificar_alertas()
                    
                    # Enviar via WebSocket
                    self.socketio.emit('atualizacao_automatica', {
                        'dados': dados,
                        'alertas': alertas,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Aguardar próxima atualização (30 segundos)
                    time.sleep(30)
                    
                except Exception as e:
                    logger.error(f"Erro nas atualizações automáticas: {e}")
                    time.sleep(60)  # Aguardar mais tempo em caso de erro
        
        # Iniciar thread de atualizações
        thread_atualizacoes = threading.Thread(target=worker_atualizacoes, daemon=True)
        thread_atualizacoes.start()
        
        logger.info("Sistema de atualizações automáticas iniciado")
    
    def parar_atualizacoes_automaticas(self):
        """Para o sistema de atualizações automáticas."""
        self.atualizacoes_ativas = False
        logger.info("Sistema de atualizações automáticas parado")
    
    def _gerar_templates(self):
        """Gera os templates HTML necessários."""
        # Template principal do dashboard
        template_dashboard = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Lotofácil - Tempo Real</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <link href="{{ url_for('static', filename='css/dashboard.css') }}" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-dark bg-primary">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-chart-line"></i> Dashboard Lotofácil
            </span>
            <div class="d-flex align-items-center">
                <span id="status-conexao" class="badge bg-success me-3">
                    <i class="fas fa-circle"></i> Online
                </span>
                <span id="ultima-atualizacao" class="text-light small"></span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-3">
        <!-- Alertas -->
        <div id="alertas-container" class="mb-3"></div>

        <!-- Cards de Estatísticas Rápidas -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card bg-primary text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h6 class="card-title">Total de Concursos</h6>
                                <h3 id="total-concursos">-</h3>
                            </div>
                            <div class="align-self-center">
                                <i class="fas fa-list-ol fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-success text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h6 class="card-title">Último Concurso</h6>
                                <h3 id="ultimo-concurso">-</h3>
                            </div>
                            <div class="align-self-center">
                                <i class="fas fa-trophy fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h6 class="card-title">Números Quentes</h6>
                                <div id="numeros-quentes" class="small">-</div>
                            </div>
                            <div class="align-self-center">
                                <i class="fas fa-fire fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h6 class="card-title">Números Frios</h6>
                                <div id="numeros-frios" class="small">-</div>
                            </div>
                            <div class="align-self-center">
                                <i class="fas fa-snowflake fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Gráficos -->
        <div class="row">
            <div class="col-lg-6 mb-4">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Frequência dos Números</h5>
                        <button class="btn btn-sm btn-outline-primary" onclick="atualizarGrafico('frequencia')">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="grafico-frequencia"></div>
                    </div>
                </div>
            </div>
            <div class="col-lg-6 mb-4">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Tendências</h5>
                        <button class="btn btn-sm btn-outline-primary" onclick="atualizarGrafico('tendencias')">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="grafico-tendencias"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-lg-6 mb-4">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Mapa de Calor</h5>
                        <button class="btn btn-sm btn-outline-primary" onclick="atualizarGrafico('heatmap')">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="grafico-heatmap"></div>
                    </div>
                </div>
            </div>
            <div class="col-lg-6 mb-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Últimos Concursos</h5>
                    </div>
                    <div class="card-body">
                        <div id="ultimos-concursos" class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Concurso</th>
                                        <th>Data</th>
                                        <th>Números</th>
                                    </tr>
                                </thead>
                                <tbody id="tabela-ultimos-concursos">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/dashboard.js') }}"></script>
</body>
</html>
        """
        
        # Salvar template
        with open("funcionalidades/templates/dashboard.html", "w", encoding="utf-8") as f:
            f.write(template_dashboard)
    
    def _gerar_arquivos_estaticos(self):
        """Gera arquivos CSS e JavaScript."""
        # CSS
        css_dashboard = """
.card {
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    border: 1px solid rgba(0, 0, 0, 0.125);
}

.card-header {
    background-color: #f8f9fa;
    border-bottom: 1px solid rgba(0, 0, 0, 0.125);
}

.badge {
    font-size: 0.75em;
}

#status-conexao.offline {
    background-color: #dc3545 !important;
}

.alert-dismissible .btn-close {
    position: absolute;
    top: 0;
    right: 0;
    z-index: 2;
    padding: 1.25rem 1rem;
}

.numero-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    margin: 0.125rem;
    background-color: #007bff;
    color: white;
    border-radius: 0.25rem;
    font-size: 0.875rem;
}

.loading {
    opacity: 0.6;
    pointer-events: none;
}

@media (max-width: 768px) {
    .container-fluid {
        padding-left: 10px;
        padding-right: 10px;
    }
    
    .card-body {
        padding: 1rem 0.5rem;
    }
}
        """
        
        with open("funcionalidades/static/css/dashboard.css", "w", encoding="utf-8") as f:
            f.write(css_dashboard)
        
        # JavaScript
        js_dashboard = """
// Configuração do Socket.IO
const socket = io();

// Variáveis globais
let dadosAtuais = {};
let graficosCarregados = {};

// Inicialização
document.addEventListener('DOMContentLoaded', function() {
    inicializarDashboard();
    configurarEventos();
});

function inicializarDashboard() {
    console.log('Inicializando dashboard...');
    
    // Carregar dados iniciais
    carregarDadosIniciais();
    
    // Configurar atualizações automáticas
    setInterval(solicitarAtualizacao, 30000); // 30 segundos
}

function configurarEventos() {
    // Eventos do Socket.IO
    socket.on('connect', function() {
        console.log('Conectado ao servidor');
        atualizarStatusConexao(true);
    });
    
    socket.on('disconnect', function() {
        console.log('Desconectado do servidor');
        atualizarStatusConexao(false);
    });
    
    socket.on('dados_iniciais', function(dados) {
        console.log('Dados iniciais recebidos:', dados);
        atualizarDashboard(dados);
    });
    
    socket.on('atualizacao_automatica', function(data) {
        console.log('Atualização automática recebida');
        atualizarDashboard(data.dados);
        
        if (data.alertas && data.alertas.total > 0) {
            exibirAlertas(data.alertas.alertas);
        }
    });
    
    socket.on('atualizacao_dados', function(data) {
        console.log('Atualização de dados:', data.tipo);
        
        if (data.tipo === 'geral') {
            atualizarDashboard(data.dados);
        }
    });
    
    socket.on('erro', function(data) {
        console.error('Erro do servidor:', data.mensagem);
        exibirAlerta('Erro: ' + data.mensagem, 'danger');
    });
}

function carregarDadosIniciais() {
    fetch('/api/dados-basicos')
        .then(response => response.json())
        .then(dados => {
            atualizarDashboard(dados);
            carregarGraficos();
        })
        .catch(error => {
            console.error('Erro ao carregar dados iniciais:', error);
            exibirAlerta('Erro ao carregar dados iniciais', 'danger');
        });
}

function atualizarDashboard(dados) {
    dadosAtuais = dados;
    
    // Atualizar cards
    document.getElementById('total-concursos').textContent = dados.total_concursos || '-';
    document.getElementById('ultimo-concurso').textContent = dados.ultimo_concurso?.numero || '-';
    
    // Números quentes
    const numerosQuentes = dados.estatisticas_rapidas?.numeros_quentes || [];
    const htmlQuentes = numerosQuentes.slice(0, 3).map(([num, freq]) => 
        `<span class="numero-badge">${num}</span>`
    ).join('');
    document.getElementById('numeros-quentes').innerHTML = htmlQuentes;
    
    // Números frios
    const numerosFrios = dados.estatisticas_rapidas?.numeros_frios || [];
    const htmlFrios = numerosFrios.slice(0, 3).map(([num, freq]) => 
        `<span class="numero-badge">${num}</span>`
    ).join('');
    document.getElementById('numeros-frios').innerHTML = htmlFrios;
    
    // Atualizar timestamp
    const agora = new Date();
    document.getElementById('ultima-atualizacao').textContent = 
        'Atualizado: ' + agora.toLocaleTimeString();
    
    // Carregar últimos concursos
    carregarUltimosConcursos();
}

function carregarGraficos() {
    atualizarGrafico('frequencia');
    atualizarGrafico('tendencias');
    atualizarGrafico('heatmap');
}

function atualizarGrafico(tipo) {
    const elemento = document.getElementById(`grafico-${tipo}`);
    if (!elemento) return;
    
    // Mostrar loading
    elemento.classList.add('loading');
    
    let url = `/api/grafico/${tipo}`;
    let params = new URLSearchParams();
    
    if (tipo === 'frequencia') {
        params.append('limite', '100');
    } else if (tipo === 'tendencias') {
        params.append('janela', '20');
        params.append('numeros', '1,5,10,15,20,25');
    } else if (tipo === 'heatmap') {
        params.append('limite', '100');
    }
    
    if (params.toString()) {
        url += '?' + params.toString();
    }
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.erro) {
                throw new Error(data.erro);
            }
            
            Plotly.newPlot(elemento, data.grafico.data, data.grafico.layout, {
                responsive: true,
                displayModeBar: false
            });
            
            graficosCarregados[tipo] = true;
        })
        .catch(error => {
            console.error(`Erro ao carregar gráfico ${tipo}:`, error);
            elemento.innerHTML = `<div class="alert alert-danger">Erro ao carregar gráfico: ${error.message}</div>`;
        })
        .finally(() => {
            elemento.classList.remove('loading');
        });
}

function carregarUltimosConcursos() {
    fetch('/api/estatisticas-tempo-real')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('tabela-ultimos-concursos');
            tbody.innerHTML = '';
            
            data.ultimos_concursos.forEach(concurso => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${concurso.concurso}</td>
                    <td>${new Date(concurso.data).toLocaleDateString('pt-BR')}</td>
                    <td>
                        ${concurso.numeros.map(n => `<span class="numero-badge">${n}</span>`).join('')}
                    </td>
                `;
            });
        })
        .catch(error => {
            console.error('Erro ao carregar últimos concursos:', error);
        });
}

function solicitarAtualizacao() {
    socket.emit('solicitar_atualizacao', { tipo: 'geral' });
}

function atualizarStatusConexao(conectado) {
    const elemento = document.getElementById('status-conexao');
    if (conectado) {
        elemento.className = 'badge bg-success me-3';
        elemento.innerHTML = '<i class="fas fa-circle"></i> Online';
    } else {
        elemento.className = 'badge bg-danger me-3';
        elemento.innerHTML = '<i class="fas fa-circle"></i> Offline';
    }
}

function exibirAlertas(alertas) {
    const container = document.getElementById('alertas-container');
    container.innerHTML = '';
    
    alertas.forEach(alerta => {
        const div = document.createElement('div');
        div.className = `alert alert-${obterCorAlerta(alerta.prioridade)} alert-dismissible fade show`;
        div.innerHTML = `
            <strong>${alerta.tipo.replace('_', ' ').toUpperCase()}:</strong> ${alerta.mensagem}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        container.appendChild(div);
    });
}

function obterCorAlerta(prioridade) {
    switch (prioridade) {
        case 'alta': return 'danger';
        case 'media': return 'warning';
        case 'baixa': return 'info';
        default: return 'secondary';
    }
}

function exibirAlerta(mensagem, tipo = 'info') {
    const container = document.getElementById('alertas-container');
    const div = document.createElement('div');
    div.className = `alert alert-${tipo} alert-dismissible fade show`;
    div.innerHTML = `
        ${mensagem}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    container.appendChild(div);
    
    // Auto-remover após 5 segundos
    setTimeout(() => {
        if (div.parentNode) {
            div.remove();
        }
    }, 5000);
}

// Funções utilitárias
function formatarNumero(numero) {
    return numero.toLocaleString('pt-BR');
}

function formatarData(data) {
    return new Date(data).toLocaleDateString('pt-BR');
}
        """
        
        with open("funcionalidades/static/js/dashboard.js", "w", encoding="utf-8") as f:
            f.write(js_dashboard)
    
    def executar(self, debug: bool = False):
        """Executa o dashboard."""
        try:
            logger.info(f"Iniciando dashboard na porta {self.port}")
            
            # Iniciar atualizações automáticas
            self.iniciar_atualizacoes_automaticas()
            
            # Executar aplicação
            self.socketio.run(self.app, 
                            host='0.0.0.0', 
                            port=self.port, 
                            debug=debug,
                            allow_unsafe_werkzeug=True)
            
        except KeyboardInterrupt:
            logger.info("Dashboard interrompido pelo usuário")
        except Exception as e:
            logger.error(f"Erro ao executar dashboard: {e}")
            raise
        finally:
            self.parar_atualizacoes_automaticas()


def exemplo_uso():
    """
    Exemplo de uso do dashboard.
    """
    # Criar e executar dashboard
    dashboard = DashboardTempoReal(port=5000)
    
    print("Dashboard Lotofácil - Tempo Real")
    print("Acesse: http://localhost:5000")
    print("Pressione Ctrl+C para parar")
    
    try:
        dashboard.executar(debug=False)
    except KeyboardInterrupt:
        print("\nDashboard finalizado.")


if __name__ == "__main__":
    exemplo_uso()