"""Módulo para integração com a API da Caixa Econômica Federal.

Este módulo fornece funcionalidades para buscar dados da Lotofácil
diretamente da API oficial da Caixa.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
try:
    from .config import api_config, get_headers, get_endpoint_url
    from .cache_service import cache_service, cached_function
except ImportError:
    # Fallback para importação absoluta
    from config import api_config, get_headers, get_endpoint_url
    from cache_service import cache_service, cached_function

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConcursoResult:
    """Classe para representar o resultado de um concurso."""
    numero: int
    data_sorteio: str
    dezenas: List[str]
    acumulado: bool
    valor_acumulado: float
    arrecadacao_total: float
    ganhadores_15_numeros: int
    ganhadores_14_numeros: int
    ganhadores_13_numeros: int
    ganhadores_12_numeros: int
    ganhadores_11_numeros: int
    valor_rateio_15_numeros: float
    valor_rateio_14_numeros: float
    valor_rateio_13_numeros: float
    valor_rateio_12_numeros: float
    valor_rateio_11_numeros: float
    proximo_concurso: int
    data_proximo_concurso: str
    valor_estimado_proximo_concurso: float

class CaixaAPI:
    """Cliente para interação com a API da Caixa Econômica Federal."""
    
    def __init__(self):
        self.config = api_config.config
        self.session = requests.Session()
        self._setup_session()
        # Usar o sistema de cache avançado
        self.cache_service = cache_service
        self.request_count = 0
        self.last_request_time = 0
        
    def _setup_session(self):
        """Configura a sessão HTTP com headers e configurações."""
        session_config = api_config.get_session_config()
        
        # Aplicar configurações da sessão
        self.session.headers.update(session_config['headers'])
        
        # Configurar adaptadores para retry automático
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info("Sessão HTTP configurada com autenticação e retry automático")
    
    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Gera chave única para cache."""
        if params:
            params_str = json.dumps(params, sort_keys=True)
            return f"caixa_api:{endpoint}:{hash(params_str)}"
        return f"caixa_api:{endpoint}"
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Recupera dados do cache se válidos."""
        if self._is_cache_valid(key):
            return self.cache[key]['data']
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Armazena dados no cache."""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _rate_limit(self):
        """Implementa rate limiting para evitar sobrecarga da API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_ultimo_concurso(self) -> Optional[ConcursoResult]:
        """Busca o resultado do último concurso da Lotofácil."""
        cache_key = self._get_cache_key('ultimo_concurso')
        
        # Verificar cache
        cached_result = self.cache_service.get(cache_key)
        if cached_result is not None:
            logger.info("Dados do último concurso obtidos do cache")
            return cached_result
        
        try:
            self._rate_limit()
            url = get_endpoint_url('lotofacil')
            logger.info(f"Buscando último concurso: {url}")
            
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            concurso = self._parse_concurso_data(data)
            
            # Armazenar no cache (TTL menor para último concurso - 5 minutos)
            if concurso:
                self.cache_service.set(cache_key, concurso, ttl=300)
            
            logger.info(f"Último concurso obtido: {data.get('numero', 'N/A')}")
            return concurso
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar último concurso: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON: {e}")
            return None
    
    def get_concurso(self, numero: int) -> Optional[ConcursoResult]:
        """Busca o resultado de um concurso específico."""
        cache_key = self._get_cache_key(f'concurso_{numero}')
        
        # Verificar cache
        cached_result = self.cache_service.get(cache_key)
        if cached_result is not None:
            logger.info(f"Dados do concurso {numero} obtidos do cache")
            return cached_result
        
        try:
            self._rate_limit()
            url = get_endpoint_url(f'lotofacil/{numero}')
            logger.info(f"Buscando concurso {numero}: {url}")
            
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Verifica se o concurso existe
            if not data or data.get('numero') != numero:
                logger.warning(f"Concurso {numero} não encontrado")
                return None
            
            concurso = self._parse_concurso_data(data)
            
            # Armazenar no cache (TTL maior para concursos específicos - 1 hora)
            if concurso:
                self.cache_service.set(cache_key, concurso, ttl=3600)
            
            logger.info(f"Concurso {numero} obtido com sucesso")
            return concurso
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar concurso {numero}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON do concurso {numero}: {e}")
            return None
    
    def get_concursos_range(self, inicio: int, fim: int) -> List[ConcursoResult]:
        """Busca uma faixa de concursos."""
        resultados = []
        
        for numero in range(inicio, fim + 1):
            resultado = self.get_concurso(numero)
            if resultado:
                resultados.append(resultado)
            else:
                logger.warning(f"Concurso {numero} não pôde ser obtido")
            
            # Rate limiting já aplicado no método get_concurso
            pass
        
        return resultados
    
    def _parse_concurso_data(self, data: Dict) -> ConcursoResult:
        """Converte os dados da API para o formato ConcursoResult."""
        try:
            return ConcursoResult(
                numero=int(data.get('numero', 0)),
                data_sorteio=data.get('dataApuracao', ''),
                dezenas=data.get('dezenas', []),
                acumulado=data.get('acumulado', False),
                valor_acumulado=float(data.get('valorAcumuladoProximoConcurso', 0)),
                arrecadacao_total=float(data.get('valorArrecadado', 0)),
                ganhadores_15_numeros=int(data.get('listaRateioPremio', [{}])[0].get('numeroDeGanhadores', 0)) if data.get('listaRateioPremio') else 0,
                ganhadores_14_numeros=int(data.get('listaRateioPremio', [{}])[1].get('numeroDeGanhadores', 0)) if len(data.get('listaRateioPremio', [])) > 1 else 0,
                ganhadores_13_numeros=int(data.get('listaRateioPremio', [{}])[2].get('numeroDeGanhadores', 0)) if len(data.get('listaRateioPremio', [])) > 2 else 0,
                ganhadores_12_numeros=int(data.get('listaRateioPremio', [{}])[3].get('numeroDeGanhadores', 0)) if len(data.get('listaRateioPremio', [])) > 3 else 0,
                ganhadores_11_numeros=int(data.get('listaRateioPremio', [{}])[4].get('numeroDeGanhadores', 0)) if len(data.get('listaRateioPremio', [])) > 4 else 0,
                valor_rateio_15_numeros=float(data.get('listaRateioPremio', [{}])[0].get('valorPremio', 0)) if data.get('listaRateioPremio') else 0,
                valor_rateio_14_numeros=float(data.get('listaRateioPremio', [{}])[1].get('valorPremio', 0)) if len(data.get('listaRateioPremio', [])) > 1 else 0,
                valor_rateio_13_numeros=float(data.get('listaRateioPremio', [{}])[2].get('valorPremio', 0)) if len(data.get('listaRateioPremio', [])) > 2 else 0,
                valor_rateio_12_numeros=float(data.get('listaRateioPremio', [{}])[3].get('valorPremio', 0)) if len(data.get('listaRateioPremio', [])) > 3 else 0,
                valor_rateio_11_numeros=float(data.get('listaRateioPremio', [{}])[4].get('valorPremio', 0)) if len(data.get('listaRateioPremio', [])) > 4 else 0,
                proximo_concurso=int(data.get('numeroConcursoProximo', 0)),
                data_proximo_concurso=data.get('dataProximoConcurso', ''),
                valor_estimado_proximo_concurso=float(data.get('valorEstimadoProximoConcurso', 0))
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Erro ao parsear dados do concurso: {e}")
            raise
    
    def clear_cache(self):
        """Limpa o cache."""
        self.cache.clear()
        logger.info("Cache limpo")
    
    def get_cache_info(self) -> Dict:
        """Retorna informações sobre o cache."""
        return {
            'total_entries': len(self.cache),
            'cache_timeout': self.config.cache_timeout,
            'entries': list(self.cache.keys())
        }
    
    def get_request_stats(self) -> Dict:
        """Retorna estatísticas de requisições."""
        return {
            'total_requests': self.request_count,
            'last_request_time': self.last_request_time,
            'rate_limit_delay': self.config.rate_limit_delay,
            'max_retries': self.config.max_retries,
            'timeout': self.config.timeout
        }
    
    def reset_stats(self):
        """Reseta as estatísticas de requisições."""
        self.request_count = 0
        self.last_request_time = 0
        logger.info("Estatísticas de requisições resetadas")
    
    def update_headers(self, additional_headers: Dict[str, str]):
        """Atualiza headers da sessão."""
        self.session.headers.update(additional_headers)
        logger.info(f"Headers atualizados: {list(additional_headers.keys())}")
    
    def test_connection(self) -> bool:
        """Testa a conexão com a API."""
        try:
            self._rate_limit()
            url = get_endpoint_url('lotofacil')
            response = self.session.head(url, timeout=self.config.timeout)
            
            if response.status_code == 200:
                logger.info("Conexão com API testada com sucesso")
                return True
            else:
                logger.warning(f"Teste de conexão retornou status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Erro no teste de conexão: {e}")
            return False
    
    def verificar_novos_concursos(self) -> List[ConcursoResult]:
        """Verifica se há novos concursos disponíveis."""
        try:
            # Por simplicidade, retorna apenas o último concurso
            ultimo = self.get_ultimo_concurso()
            if ultimo:
                logger.info(f"Último concurso disponível: {ultimo.numero}")
                return [ultimo]
            return []
        except Exception as e:
            logger.error(f"Erro ao verificar novos concursos: {e}")
            return []
    
    def salvar_concursos(self, concursos: List[ConcursoResult]):
        """Salva concursos no banco de dados."""
        try:
            # Importa aqui para evitar dependência circular
            import sqlite3
            import os
            
            # Garante que o diretório existe
            os.makedirs('dados', exist_ok=True)
            
            conn = sqlite3.connect('dados/lotofacil.db')
            cursor = conn.cursor()
            
            # Cria a tabela se não existir
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS concursos (
                    numero INTEGER PRIMARY KEY,
                    data_sorteio TEXT,
                    dezenas TEXT,
                    acumulado INTEGER,
                    valor_acumulado REAL,
                    arrecadacao_total REAL,
                    ganhadores_15 INTEGER,
                    ganhadores_14 INTEGER,
                    ganhadores_13 INTEGER,
                    ganhadores_12 INTEGER,
                    ganhadores_11 INTEGER,
                    valor_15 REAL,
                    valor_14 REAL,
                    valor_13 REAL,
                    valor_12 REAL,
                    valor_11 REAL
                )
            ''')
            
            for concurso in concursos:
                cursor.execute('''
                    INSERT OR REPLACE INTO concursos VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    concurso.numero,
                    concurso.data_sorteio,
                    ','.join(concurso.dezenas),
                    1 if concurso.acumulado else 0,
                    concurso.valor_acumulado,
                    concurso.arrecadacao_total,
                    concurso.ganhadores_15_numeros,
                    concurso.ganhadores_14_numeros,
                    concurso.ganhadores_13_numeros,
                    concurso.ganhadores_12_numeros,
                    concurso.ganhadores_11_numeros,
                    concurso.valor_rateio_15_numeros,
                    concurso.valor_rateio_14_numeros,
                    concurso.valor_rateio_13_numeros,
                    concurso.valor_rateio_12_numeros,
                    concurso.valor_rateio_11_numeros
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Salvos {len(concursos)} concursos no banco de dados")
            
        except Exception as e:
            logger.error(f"Erro ao salvar concursos: {e}")
            raise

# Instância global da API
caixa_api = CaixaAPI()

# Funções de conveniência
def get_ultimo_resultado() -> Optional[ConcursoResult]:
    """Função de conveniência para obter o último resultado."""
    return caixa_api.get_ultimo_concurso()

def get_resultado_concurso(numero: int) -> Optional[ConcursoResult]:
    """Função de conveniência para obter resultado de um concurso específico."""
    return caixa_api.get_concurso(numero)

def get_resultados_range(inicio: int, fim: int) -> List[ConcursoResult]:
    """Função de conveniência para obter uma faixa de resultados."""
    return caixa_api.get_concursos_range(inicio, fim)