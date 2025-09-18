"""Módulo API do Sistema Lotofácil.

Este módulo contém todas as funcionalidades relacionadas à API,
incluindo integração com a Caixa, cache, predições e serviços em tempo real.
"""

# Importações básicas primeiro
from .config import *
from .error_handler import *

# Depois as classes de dados
from .caixa_api import ConcursoResult

# Em seguida os serviços
from .cache_service import cache_service, MultiLevelCache
from .caixa_api import CaixaAPI
from .realtime_service import RealtimeService
from .auto_update import AutoUpdateService

# Por último as APIs
from .api_predicoes import *

__all__ = [
    'ConcursoResult',
    'cache_service',
    'MultiLevelCache', 
    'CaixaAPI',
    'RealtimeService',
    'AutoUpdateService'
]