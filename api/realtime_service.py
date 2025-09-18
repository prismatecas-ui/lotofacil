"""Serviço para busca de resultados em tempo real da Lotofácil.

Este módulo fornece funcionalidades avançadas para monitoramento
e busca de resultados da Lotofácil em tempo real.
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .caixa_api import CaixaAPI, ConcursoResult
from .config import api_config

logger = logging.getLogger(__name__)

@dataclass
class RealtimeConfig:
    """Configurações para busca em tempo real."""
    check_interval: int = 60  # segundos
    max_concurrent_requests: int = 5
    enable_notifications: bool = True
    auto_update_database: bool = True
    retry_failed_requests: bool = True
    max_retry_attempts: int = 3

@dataclass
class UpdateEvent:
    """Evento de atualização de concurso."""
    concurso: ConcursoResult
    timestamp: datetime
    event_type: str  # 'new', 'updated', 'error'
    source: str = 'api'

class RealtimeMonitor:
    """Monitor para atualizações em tempo real."""
    
    def __init__(self, config: Optional[RealtimeConfig] = None):
        self.config = config or RealtimeConfig()
        self.api = CaixaAPI()
        self.is_monitoring = False
        self.last_check = None
        self.callbacks: List[Callable[[UpdateEvent], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        
    def add_callback(self, callback: Callable[[UpdateEvent], None]):
        """Adiciona callback para eventos de atualização."""
        self.callbacks.append(callback)
        logger.info(f"Callback adicionado: {callback.__name__}")
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Adiciona callback para eventos de erro."""
        self.error_callbacks.append(callback)
        logger.info(f"Error callback adicionado: {callback.__name__}")
    
    def _notify_callbacks(self, event: UpdateEvent):
        """Notifica todos os callbacks registrados."""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Erro no callback {callback.__name__}: {e}")
    
    def _notify_error_callbacks(self, error: Exception):
        """Notifica callbacks de erro."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Erro no error callback {callback.__name__}: {e}")
    
    def check_for_updates(self) -> List[UpdateEvent]:
        """Verifica por atualizações de concursos."""
        events = []
        
        try:
            # Buscar último concurso
            ultimo_concurso = self.api.get_ultimo_concurso()
            
            if ultimo_concurso:
                event = UpdateEvent(
                    concurso=ultimo_concurso,
                    timestamp=datetime.now(),
                    event_type='new' if self.last_check is None else 'updated'
                )
                events.append(event)
                
                # Notificar callbacks
                self._notify_callbacks(event)
                
                logger.info(f"Concurso {ultimo_concurso.numero} verificado")
            
            self.last_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Erro ao verificar atualizações: {e}")
            self._notify_error_callbacks(e)
            
            error_event = UpdateEvent(
                concurso=None,
                timestamp=datetime.now(),
                event_type='error'
            )
            events.append(error_event)
        
        return events
    
    def start_monitoring(self):
        """Inicia o monitoramento em tempo real."""
        if self.is_monitoring:
            logger.warning("Monitoramento já está ativo")
            return
        
        self.is_monitoring = True
        logger.info(f"Iniciando monitoramento (intervalo: {self.config.check_interval}s)")
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    self.check_for_updates()
                    time.sleep(self.config.check_interval)
                except KeyboardInterrupt:
                    logger.info("Monitoramento interrompido pelo usuário")
                    break
                except Exception as e:
                    logger.error(f"Erro no loop de monitoramento: {e}")
                    self._notify_error_callbacks(e)
                    time.sleep(self.config.check_interval)
        
        # Executar em thread separada
        self.executor.submit(monitor_loop)
    
    def stop_monitoring(self):
        """Para o monitoramento."""
        self.is_monitoring = False
        logger.info("Monitoramento parado")
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status do monitor."""
        return {
            'is_monitoring': self.is_monitoring,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'config': asdict(self.config),
            'callbacks_count': len(self.callbacks),
            'error_callbacks_count': len(self.error_callbacks)
        }

class RealtimeService:
    """Serviço principal para operações em tempo real."""
    
    def __init__(self):
        self.api = CaixaAPI()
        self.monitor = RealtimeMonitor()
        self.session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Obtém sessão aiohttp configurada."""
        if self.session is None or self.session.closed:
            headers = api_config.get_headers()
            timeout = aiohttp.ClientTimeout(total=api_config.config.timeout)
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10)
            )
        
        return self.session
    
    async def get_concurso_async(self, numero: int) -> Optional[ConcursoResult]:
        """Busca concurso de forma assíncrona."""
        try:
            session = await self._get_session()
            url = api_config.get_endpoint_url(f'lotofacil/{numero}')
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self.api._parse_concurso_data(data)
                else:
                    logger.warning(f"Status {response.status} para concurso {numero}")
                    return None
                    
        except Exception as e:
            logger.error(f"Erro ao buscar concurso {numero} async: {e}")
            return None
    
    async def get_multiple_concursos_async(self, numeros: List[int]) -> List[Optional[ConcursoResult]]:
        """Busca múltiplos concursos de forma assíncrona."""
        tasks = [self.get_concurso_async(numero) for numero in numeros]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar exceções
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Erro ao buscar concurso {numeros[i]}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_concursos_parallel(self, numeros: List[int]) -> List[Optional[ConcursoResult]]:
        """Busca múltiplos concursos em paralelo usando threads."""
        with ThreadPoolExecutor(max_workers=api_config.config.max_retries) as executor:
            future_to_numero = {executor.submit(self.api.get_concurso, numero): numero for numero in numeros}
            results = {}
            
            for future in as_completed(future_to_numero):
                numero = future_to_numero[future]
                try:
                    result = future.result()
                    results[numero] = result
                except Exception as e:
                    logger.error(f"Erro ao buscar concurso {numero}: {e}")
                    results[numero] = None
        
        # Retornar na ordem original
        return [results.get(numero) for numero in numeros]
    
    def get_latest_results(self, count: int = 10) -> List[ConcursoResult]:
        """Busca os últimos resultados disponíveis."""
        try:
            # Primeiro, buscar o último concurso para saber o número atual
            ultimo = self.api.get_ultimo_concurso()
            if not ultimo:
                logger.error("Não foi possível obter o último concurso")
                return []
            
            # Calcular range de concursos
            inicio = max(1, ultimo.numero - count + 1)
            fim = ultimo.numero
            
            logger.info(f"Buscando concursos {inicio} a {fim}")
            
            # Buscar em paralelo
            numeros = list(range(inicio, fim + 1))
            resultados = self.get_concursos_parallel(numeros)
            
            # Filtrar resultados válidos
            resultados_validos = [r for r in resultados if r is not None]
            
            logger.info(f"Obtidos {len(resultados_validos)} resultados válidos")
            return resultados_validos
            
        except Exception as e:
            logger.error(f"Erro ao buscar últimos resultados: {e}")
            return []
    
    def search_concursos_by_date(self, data_inicio: datetime, data_fim: datetime) -> List[ConcursoResult]:
        """Busca concursos por período de datas."""
        # Esta é uma implementação simplificada
        # Em um cenário real, seria necessário um mapeamento de datas para números de concursos
        logger.warning("Busca por data não implementada completamente - usando últimos 50 concursos")
        return self.get_latest_results(50)
    
    async def close(self):
        """Fecha recursos assíncronos."""
        if self.session and not self.session.closed:
            await self.session.close()
        
        self.monitor.stop_monitoring()
        logger.info("Serviço realtime fechado")
    
    def __del__(self):
        """Destrutor para limpeza."""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            # Não podemos usar await aqui, então apenas logamos
            logger.warning("Sessão aiohttp não foi fechada adequadamente")

# Instância global do serviço
realtime_service = RealtimeService()

# Funções de conveniência
def get_latest_results(count: int = 10) -> List[ConcursoResult]:
    """Função de conveniência para obter últimos resultados."""
    return realtime_service.get_latest_results(count)

def start_monitoring(callback: Optional[Callable[[UpdateEvent], None]] = None) -> RealtimeMonitor:
    """Inicia monitoramento em tempo real."""
    monitor = realtime_service.monitor
    if callback:
        monitor.add_callback(callback)
    monitor.start_monitoring()
    return monitor

def stop_monitoring():
    """Para o monitoramento."""
    realtime_service.monitor.stop_monitoring()

async def get_concursos_async(numeros: List[int]) -> List[Optional[ConcursoResult]]:
    """Função assíncrona para buscar múltiplos concursos."""
    return await realtime_service.get_multiple_concursos_async(numeros)