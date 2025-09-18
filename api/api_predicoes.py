from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import tensorflow as tf
from datetime import datetime, timedelta
import json
import os
import sys
import logging
from pathlib import Path
import joblib
import threading
import time
from collections import deque
import sqlite3
from functools import wraps
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
from enum import Enum

# Adicionar diretório pai ao path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from modelo.modelo_tensorflow2 import LotofacilModel
    from modelo.algoritmos_avancados import AlgoritmosAvancados
    from modelo.analise_padroes import AnalisePadroesLotofacil
    from dados.dados import carregar_dados, preparar_dados
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que os módulos estão no diretório correto")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_predicoes.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enums
class TipoModelo(Enum):
    TENSORFLOW = "tensorflow"
    ALGORITMOS_AVANCADOS = "algoritmos_avancados"
    ANALISE_PADROES = "analise_padroes"
    ENSEMBLE = "ensemble"

class StatusPredicao(Enum):
    SUCESSO = "sucesso"
    ERRO = "erro"
    PROCESSANDO = "processando"

@dataclass
class ResultadoPredicao:
    numeros_preditos: List[int]
    probabilidades: List[float]
    confianca: float
    modelo_usado: str
    timestamp: datetime
    tempo_processamento: float
    metadados: Dict[str, Any]

class CachePredicoes:
    """
    Sistema de cache para predições
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def _generate_key(self, dados_entrada: Dict[str, Any], modelo: str) -> str:
        """
        Gera chave única para o cache
        """
        key_data = f"{json.dumps(dados_entrada, sort_keys=True)}_{modelo}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, dados_entrada: Dict[str, Any], modelo: str) -> Optional[ResultadoPredicao]:
        """
        Recupera predição do cache
        """
        with self.lock:
            key = self._generate_key(dados_entrada, modelo)
            
            if key in self.cache:
                # Verificar TTL
                if time.time() - self.timestamps[key] < self.ttl_seconds:
                    return self.cache[key]
                else:
                    # Remover entrada expirada
                    del self.cache[key]
                    del self.timestamps[key]
            
            return None
    
    def set(self, dados_entrada: Dict[str, Any], modelo: str, resultado: ResultadoPredicao):
        """
        Armazena predição no cache
        """
        with self.lock:
            key = self._generate_key(dados_entrada, modelo)
            
            # Limpar cache se necessário
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = resultado
            self.timestamps[key] = time.time()
    
    def clear(self):
        """
        Limpa todo o cache
        """
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

class GerenciadorModelos:
    """
    Gerencia carregamento e uso de múltiplos modelos
    """
    
    def __init__(self):
        self.modelos = {}
        self.modelos_carregados = {}
        self.ultimo_carregamento = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    def registrar_modelo(self, nome: str, tipo: TipoModelo, caminho: Optional[str] = None):
        """
        Registra um modelo para uso
        """
        self.modelos[nome] = {
            'tipo': tipo,
            'caminho': caminho,
            'ativo': True
        }
        logger.info(f"Modelo {nome} registrado: {tipo.value}")
    
    def carregar_modelo(self, nome: str) -> Any:
        """
        Carrega modelo na memória
        """
        with self.lock:
            if nome not in self.modelos:
                raise ValueError(f"Modelo {nome} não registrado")
            
            # Verificar se já está carregado
            if nome in self.modelos_carregados:
                return self.modelos_carregados[nome]
            
            config = self.modelos[nome]
            tipo = config['tipo']
            
            try:
                if tipo == TipoModelo.TENSORFLOW:
                    modelo = LotofacilModel()
                    if config['caminho'] and os.path.exists(config['caminho']):
                        modelo.carregar_modelo(config['caminho'])
                    else:
                        logger.warning(f"Modelo TensorFlow {nome} não encontrado, criando novo")
                        modelo.criar_modelo()
                
                elif tipo == TipoModelo.ALGORITMOS_AVANCADOS:
                    modelo = AlgoritmosAvancados()
                
                elif tipo == TipoModelo.ANALISE_PADROES:
                    modelo = AnalisePadroesLotofacil()
                
                else:
                    raise ValueError(f"Tipo de modelo não suportado: {tipo}")
                
                self.modelos_carregados[nome] = modelo
                self.ultimo_carregamento[nome] = datetime.now()
                
                logger.info(f"Modelo {nome} carregado com sucesso")
                return modelo
            
            except Exception as e:
                logger.error(f"Erro ao carregar modelo {nome}: {e}")
                raise
    
    def obter_modelo(self, nome: str) -> Any:
        """
        Obtém modelo (carrega se necessário)
        """
        if nome not in self.modelos_carregados:
            return self.carregar_modelo(nome)
        return self.modelos_carregados[nome]
    
    def listar_modelos(self) -> Dict[str, Dict[str, Any]]:
        """
        Lista todos os modelos registrados
        """
        resultado = {}
        for nome, config in self.modelos.items():
            resultado[nome] = {
                'tipo': config['tipo'].value,
                'ativo': config['ativo'],
                'carregado': nome in self.modelos_carregados,
                'ultimo_carregamento': self.ultimo_carregamento.get(nome)
            }
        return resultado

class APIPredicoes:
    """
    API principal para predições em tempo real
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.gerenciador_modelos = GerenciadorModelos()
        self.cache = CachePredicoes()
        self.historico_predicoes = deque(maxlen=1000)
        self.estatisticas = {
            'total_predicoes': 0,
            'predicoes_por_modelo': {},
            'tempo_medio_resposta': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self._configurar_rotas()
        self._inicializar_modelos()
    
    def _inicializar_modelos(self):
        """
        Inicializa modelos padrão
        """
        try:
            # Registrar modelos disponíveis
            self.gerenciador_modelos.registrar_modelo(
                "tensorflow_basico", 
                TipoModelo.TENSORFLOW
            )
            
            self.gerenciador_modelos.registrar_modelo(
                "algoritmos_avancados", 
                TipoModelo.ALGORITMOS_AVANCADOS
            )
            
            self.gerenciador_modelos.registrar_modelo(
                "analise_padroes", 
                TipoModelo.ANALISE_PADROES
            )
            
            logger.info("Modelos inicializados com sucesso")
        
        except Exception as e:
            logger.error(f"Erro ao inicializar modelos: {e}")
    
    def _configurar_rotas(self):
        """
        Configura todas as rotas da API
        """
        
        @self.app.route('/', methods=['GET'])
        def home():
            return render_template_string(self._get_home_template())
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'modelos_carregados': len(self.gerenciador_modelos.modelos_carregados),
                'cache_size': len(self.cache.cache)
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            return self._handle_prediction()
        
        @self.app.route('/predict/batch', methods=['POST'])
        def predict_batch():
            return self._handle_batch_prediction()
        
        @self.app.route('/models', methods=['GET'])
        def list_models():
            return jsonify(self.gerenciador_modelos.listar_modelos())
        
        @self.app.route('/models/<nome>/load', methods=['POST'])
        def load_model(nome):
            try:
                self.gerenciador_modelos.carregar_modelo(nome)
                return jsonify({'status': 'success', 'message': f'Modelo {nome} carregado'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 400
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            return jsonify(self.estatisticas)
        
        @self.app.route('/cache/clear', methods=['POST'])
        def clear_cache():
            self.cache.clear()
            return jsonify({'status': 'success', 'message': 'Cache limpo'})
        
        @self.app.route('/history', methods=['GET'])
        def get_history():
            limit = request.args.get('limit', 10, type=int)
            history = list(self.historico_predicoes)[-limit:]
            return jsonify([self._serialize_resultado(r) for r in history])
    
    def _handle_prediction(self) -> Dict[str, Any]:
        """
        Processa uma predição individual
        """
        try:
            dados = request.get_json()
            
            if not dados:
                return jsonify({'error': 'Dados não fornecidos'}), 400
            
            # Validar entrada
            erro_validacao = self._validar_entrada_predicao(dados)
            if erro_validacao:
                return jsonify({'error': erro_validacao}), 400
            
            # Extrair parâmetros
            numeros_entrada = dados.get('numeros', [])
            modelo_nome = dados.get('modelo', 'tensorflow_basico')
            usar_cache = dados.get('cache', True)
            
            # Verificar cache
            if usar_cache:
                resultado_cache = self.cache.get(dados, modelo_nome)
                if resultado_cache:
                    self.estatisticas['cache_hits'] += 1
                    return jsonify(self._serialize_resultado(resultado_cache))
            
            self.estatisticas['cache_misses'] += 1
            
            # Fazer predição
            inicio = time.time()
            resultado = self._executar_predicao(numeros_entrada, modelo_nome, dados)
            tempo_processamento = time.time() - inicio
            
            resultado.tempo_processamento = tempo_processamento
            
            # Atualizar estatísticas
            self._atualizar_estatisticas(modelo_nome, tempo_processamento)
            
            # Salvar no cache
            if usar_cache:
                self.cache.set(dados, modelo_nome, resultado)
            
            # Adicionar ao histórico
            self.historico_predicoes.append(resultado)
            
            return jsonify(self._serialize_resultado(resultado))
        
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return jsonify({'error': f'Erro interno: {str(e)}'}), 500
    
    def _handle_batch_prediction(self) -> Dict[str, Any]:
        """
        Processa múltiplas predições
        """
        try:
            dados = request.get_json()
            
            if not dados or 'predicoes' not in dados:
                return jsonify({'error': 'Lista de predições não fornecida'}), 400
            
            predicoes = dados['predicoes']
            if len(predicoes) > 100:  # Limite de segurança
                return jsonify({'error': 'Máximo 100 predições por batch'}), 400
            
            resultados = []
            
            for i, predicao in enumerate(predicoes):
                try:
                    # Validar entrada
                    erro_validacao = self._validar_entrada_predicao(predicao)
                    if erro_validacao:
                        resultados.append({
                            'indice': i,
                            'status': 'erro',
                            'erro': erro_validacao
                        })
                        continue
                    
                    # Executar predição
                    numeros_entrada = predicao.get('numeros', [])
                    modelo_nome = predicao.get('modelo', 'tensorflow_basico')
                    
                    resultado = self._executar_predicao(numeros_entrada, modelo_nome, predicao)
                    
                    resultados.append({
                        'indice': i,
                        'status': 'sucesso',
                        'resultado': self._serialize_resultado(resultado)
                    })
                
                except Exception as e:
                    resultados.append({
                        'indice': i,
                        'status': 'erro',
                        'erro': str(e)
                    })
            
            return jsonify({
                'total_predicoes': len(predicoes),
                'sucessos': len([r for r in resultados if r['status'] == 'sucesso']),
                'erros': len([r for r in resultados if r['status'] == 'erro']),
                'resultados': resultados
            })
        
        except Exception as e:
            logger.error(f"Erro no batch: {e}")
            return jsonify({'error': f'Erro interno: {str(e)}'}), 500
    
    def _validar_entrada_predicao(self, dados: Dict[str, Any]) -> Optional[str]:
        """
        Valida dados de entrada para predição
        """
        
        if 'numeros' in dados:
            numeros = dados['numeros']
            
            if not isinstance(numeros, list):
                return "'numeros' deve ser uma lista"
            
            if len(numeros) != 15:
                return "Deve fornecer exatamente 15 números"
            
            for num in numeros:
                if not isinstance(num, int) or num < 1 or num > 25:
                    return "Números devem ser inteiros entre 1 e 25"
            
            if len(set(numeros)) != 15:
                return "Números não podem se repetir"
        
        if 'modelo' in dados:
            modelo = dados['modelo']
            if modelo not in self.gerenciador_modelos.modelos:
                return f"Modelo '{modelo}' não disponível"
        
        return None
    
    def _executar_predicao(self, numeros_entrada: List[int], modelo_nome: str, dados_completos: Dict[str, Any]) -> ResultadoPredicao:
        """
        Executa a predição usando o modelo especificado
        """
        
        try:
            modelo = self.gerenciador_modelos.obter_modelo(modelo_nome)
            
            # Preparar entrada
            if numeros_entrada:
                # Converter para vetor binário
                vetor_entrada = [0] * 25
                for num in numeros_entrada:
                    vetor_entrada[num - 1] = 1
                
                entrada = np.array([vetor_entrada])
            else:
                # Usar dados históricos se disponível
                entrada = self._preparar_entrada_historica()
            
            # Fazer predição baseada no tipo de modelo
            config_modelo = self.gerenciador_modelos.modelos[modelo_nome]
            tipo_modelo = config_modelo['tipo']
            
            if tipo_modelo == TipoModelo.TENSORFLOW:
                predicao = modelo.predizer(entrada)
                if isinstance(predicao, tuple):
                    probabilidades = predicao[0][0]
                else:
                    probabilidades = predicao[0]
            
            elif tipo_modelo == TipoModelo.ALGORITMOS_AVANCADOS:
                # Usar ensemble do modelo avançado
                probabilidades = modelo.predizer_ensemble(entrada)[0]
            
            elif tipo_modelo == TipoModelo.ANALISE_PADROES:
                # Usar predição baseada em padrões
                numeros_preditos = modelo.predizer_numeros()
                probabilidades = [0.0] * 25
                for num in numeros_preditos:
                    if 1 <= num <= 25:
                        probabilidades[num - 1] = 0.8  # Confiança alta para padrões
            
            else:
                raise ValueError(f"Tipo de modelo não implementado: {tipo_modelo}")
            
            # Selecionar top 15 números
            indices_ordenados = np.argsort(probabilidades)[::-1]
            numeros_preditos = [i + 1 for i in indices_ordenados[:15]]
            probs_selecionadas = [float(probabilidades[i]) for i in indices_ordenados[:15]]
            
            # Calcular confiança
            confianca = float(np.mean(probs_selecionadas))
            
            return ResultadoPredicao(
                numeros_preditos=numeros_preditos,
                probabilidades=probs_selecionadas,
                confianca=confianca,
                modelo_usado=modelo_nome,
                timestamp=datetime.now(),
                tempo_processamento=0.0,  # Será definido depois
                metadados={
                    'tipo_modelo': tipo_modelo.value,
                    'entrada_fornecida': bool(numeros_entrada),
                    'parametros_extras': {k: v for k, v in dados_completos.items() 
                                        if k not in ['numeros', 'modelo', 'cache']}
                }
            )
        
        except Exception as e:
            logger.error(f"Erro na execução da predição: {e}")
            raise
    
    def _preparar_entrada_historica(self) -> np.ndarray:
        """
        Prepara entrada baseada em dados históricos
        """
        try:
            # Tentar carregar dados históricos
            dados = carregar_dados()
            if dados is not None and len(dados) > 0:
                # Usar último sorteio como entrada
                ultimo_sorteio = dados.iloc[-1]
                
                # Extrair números do último sorteio
                numeros = []
                for i in range(1, 16):
                    col_name = f'Bola{i:02d}'
                    if col_name in ultimo_sorteio:
                        numeros.append(int(ultimo_sorteio[col_name]))
                
                if len(numeros) == 15:
                    vetor = [0] * 25
                    for num in numeros:
                        if 1 <= num <= 25:
                            vetor[num - 1] = 1
                    return np.array([vetor])
            
            # Fallback: entrada aleatória
            logger.warning("Usando entrada aleatória como fallback")
            vetor = [0] * 25
            numeros_aleatorios = np.random.choice(range(1, 26), 15, replace=False)
            for num in numeros_aleatorios:
                vetor[num - 1] = 1
            
            return np.array([vetor])
        
        except Exception as e:
            logger.error(f"Erro ao preparar entrada histórica: {e}")
            # Entrada padrão em caso de erro
            vetor = [0] * 25
            for i in range(15):
                vetor[i] = 1
            return np.array([vetor])
    
    def _atualizar_estatisticas(self, modelo_nome: str, tempo_processamento: float):
        """
        Atualiza estatísticas da API
        """
        self.estatisticas['total_predicoes'] += 1
        
        if modelo_nome not in self.estatisticas['predicoes_por_modelo']:
            self.estatisticas['predicoes_por_modelo'][modelo_nome] = 0
        self.estatisticas['predicoes_por_modelo'][modelo_nome] += 1
        
        # Atualizar tempo médio (média móvel)
        total = self.estatisticas['total_predicoes']
        tempo_atual = self.estatisticas['tempo_medio_resposta']
        self.estatisticas['tempo_medio_resposta'] = (
            (tempo_atual * (total - 1) + tempo_processamento) / total
        )
    
    def _serialize_resultado(self, resultado: ResultadoPredicao) -> Dict[str, Any]:
        """
        Serializa resultado para JSON
        """
        return {
            'numeros_preditos': resultado.numeros_preditos,
            'probabilidades': resultado.probabilidades,
            'confianca': resultado.confianca,
            'modelo_usado': resultado.modelo_usado,
            'timestamp': resultado.timestamp.isoformat(),
            'tempo_processamento': resultado.tempo_processamento,
            'metadados': resultado.metadados
        }
    
    def _get_home_template(self) -> str:
        """
        Template HTML para página inicial
        """
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Predições Lotofácil</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .method { color: #fff; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
                .get { background: #61affe; }
                .post { background: #49cc90; }
                code { background: #f8f8f8; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>API Predições Lotofácil - Tempo Real</h1>
            <p>API para predições em tempo real dos números da Lotofácil usando múltiplos modelos de machine learning.</p>
            
            <h2>Endpoints Disponíveis:</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/health</code>
                <p>Verifica status da API</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <code>/predict</code>
                <p>Faz uma predição individual</p>
                <pre>{
  "numeros": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "modelo": "tensorflow_basico",
  "cache": true
}</pre>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <code>/predict/batch</code>
                <p>Faz múltiplas predições</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/models</code>
                <p>Lista modelos disponíveis</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/stats</code>
                <p>Estatísticas da API</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/history</code>
                <p>Histórico de predições</p>
            </div>
            
            <h2>Modelos Disponíveis:</h2>
            <ul>
                <li><strong>tensorflow_basico</strong>: Modelo neural básico com TensorFlow</li>
                <li><strong>algoritmos_avancados</strong>: Ensemble com Random Forest, XGBoost e redes neurais</li>
                <li><strong>analise_padroes</strong>: Análise de padrões e frequências</li>
            </ul>
        </body>
        </html>
        """
    
    def executar(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """
        Executa a API
        """
        logger.info(f"Iniciando API de Predições na porta {port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


def criar_api() -> APIPredicoes:
    """
    Factory function para criar instância da API
    """
    return APIPredicoes()


if __name__ == "__main__":
    # Executar API
    api = criar_api()
    
    # Configurações via argumentos ou variáveis de ambiente
    import argparse
    
    parser = argparse.ArgumentParser(description='API Predições Lotofácil')
    parser.add_argument('--host', default='0.0.0.0', help='Host da API')
    parser.add_argument('--port', type=int, default=5000, help='Porta da API')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    try:
        api.executar(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("API interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro ao executar API: {e}")