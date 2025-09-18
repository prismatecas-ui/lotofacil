"""Sistema de tratamento de erros e fallbacks para API da Lotofácil.

Este módulo implementa estratégias robustas de tratamento de erros,
recuperação automática e fallbacks para garantir a confiabilidade do sistema.
"""

import asyncio
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import requests
from requests.exceptions import (
    RequestException, ConnectionError, Timeout, HTTPError,
    TooManyRedirects, URLRequired, InvalidURL
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from .caixa_api import ConcursoResult

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Níveis de severidade de erro."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categorias de erro."""
    NETWORK = "network"
    API = "api"
    DATABASE = "database"
    VALIDATION = "validation"
    SYSTEM = "system"
    CACHE = "cache"
    TIMEOUT = "timeout"

@dataclass
class ErrorInfo:
    """Informações detalhadas sobre um erro."""
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    traceback_info: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'traceback_info': self.traceback_info,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }

@dataclass
class RetryConfig:
    """Configuração de retry."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        ConnectionError, Timeout, HTTPError, OperationalError
    ])

class ErrorHandler:
    """Gerenciador principal de erros e fallbacks."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.errors: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
        self.fallback_data: Dict[str, Any] = {}
        self.log_file = log_file or "logs/error_log.json"
        self.setup_logging()
    
    def setup_logging(self):
        """Configura logging de erros."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(exist_ok=True)
        
        # Configurar handler para arquivo
        file_handler = logging.FileHandler(log_path.with_suffix('.log'), encoding='utf-8')
        file_handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Classifica um erro e retorna informações detalhadas."""
        context = context or {}
        error_type = type(error).__name__
        message = str(error)
        
        # Classificar categoria
        if isinstance(error, (ConnectionError, Timeout, TooManyRedirects)):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
        elif isinstance(error, HTTPError):
            category = ErrorCategory.API
            severity = ErrorSeverity.HIGH if error.response.status_code >= 500 else ErrorSeverity.MEDIUM
        elif isinstance(error, SQLAlchemyError):
            category = ErrorCategory.DATABASE
            severity = ErrorSeverity.HIGH if isinstance(error, OperationalError) else ErrorSeverity.MEDIUM
        elif isinstance(error, (ValueError, TypeError)):
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.LOW
        elif isinstance(error, TimeoutError):
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.MEDIUM
        else:
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.HIGH
        
        return ErrorInfo(
            error_type=error_type,
            message=message,
            category=category,
            severity=severity,
            timestamp=datetime.now(),
            context=context,
            traceback_info=traceback.format_exc()
        )
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Processa e registra um erro."""
        error_info = self.classify_error(error, context)
        
        # Registrar erro
        self.errors.append(error_info)
        self.error_counts[error_info.error_type] = self.error_counts.get(error_info.error_type, 0) + 1
        self.last_errors[error_info.error_type] = error_info.timestamp
        
        # Log baseado na severidade
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"ERRO CRÍTICO: {error_info.message}", extra={'context': context})
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(f"ERRO: {error_info.message}", extra={'context': context})
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"AVISO: {error_info.message}", extra={'context': context})
        else:
            logger.info(f"INFO: {error_info.message}", extra={'context': context})
        
        # Salvar em arquivo JSON
        self._save_error_to_file(error_info)
        
        return error_info
    
    def _save_error_to_file(self, error_info: ErrorInfo):
        """Salva erro em arquivo JSON."""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(exist_ok=True)
            
            # Carregar erros existentes
            errors_data = []
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        errors_data = json.load(f)
                except json.JSONDecodeError:
                    errors_data = []
            
            # Adicionar novo erro
            errors_data.append(error_info.to_dict())
            
            # Manter apenas os últimos 1000 erros
            if len(errors_data) > 1000:
                errors_data = errors_data[-1000:]
            
            # Salvar
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(errors_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar log de erro: {e}")
    
    def should_retry(self, error_info: ErrorInfo, retry_config: RetryConfig) -> bool:
        """Determina se deve tentar novamente."""
        if error_info.retry_count >= retry_config.max_retries:
            return False
        
        # Verificar se o tipo de erro permite retry
        error_class = globals().get(error_info.error_type)
        if error_class and not any(issubclass(error_class, exc) for exc in retry_config.retry_on_exceptions):
            return False
        
        # Verificar frequência de erros
        error_type = error_info.error_type
        if error_type in self.error_counts and self.error_counts[error_type] > 10:
            # Muitos erros do mesmo tipo, aguardar mais tempo
            last_error = self.last_errors.get(error_type)
            if last_error and (datetime.now() - last_error).seconds < 300:  # 5 minutos
                return False
        
        return True
    
    def calculate_retry_delay(self, retry_count: int, retry_config: RetryConfig) -> float:
        """Calcula delay para próxima tentativa."""
        delay = retry_config.base_delay * (retry_config.exponential_base ** retry_count)
        delay = min(delay, retry_config.max_delay)
        
        if retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Jitter de 50-100%
        
        return delay
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de erros."""
        total_errors = len(self.errors)
        if total_errors == 0:
            return {'total_errors': 0}
        
        # Estatísticas por categoria
        category_stats = {}
        for error in self.errors:
            cat = error.category.value
            category_stats[cat] = category_stats.get(cat, 0) + 1
        
        # Estatísticas por severidade
        severity_stats = {}
        for error in self.errors:
            sev = error.severity.value
            severity_stats[sev] = severity_stats.get(sev, 0) + 1
        
        # Erros mais frequentes
        most_frequent = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Erros recentes (última hora)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_errors = [e for e in self.errors if e.timestamp > recent_cutoff]
        
        return {
            'total_errors': total_errors,
            'category_stats': category_stats,
            'severity_stats': severity_stats,
            'most_frequent_errors': most_frequent,
            'recent_errors_count': len(recent_errors),
            'error_rate_last_hour': len(recent_errors)
        }

class FallbackManager:
    """Gerenciador de fallbacks e dados de backup."""
    
    def __init__(self, backup_dir: str = "backup"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.fallback_sources = []
    
    def add_fallback_source(self, name: str, source_func: Callable, priority: int = 1):
        """Adiciona uma fonte de fallback."""
        self.fallback_sources.append({
            'name': name,
            'func': source_func,
            'priority': priority
        })
        # Ordenar por prioridade
        self.fallback_sources.sort(key=lambda x: x['priority'])
        logger.info(f"Fonte de fallback adicionada: {name} (prioridade: {priority})")
    
    def save_backup_data(self, data: Any, backup_name: str):
        """Salva dados de backup."""
        try:
            backup_file = self.backup_dir / f"{backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                if isinstance(data, list) and data and hasattr(data[0], '__dict__'):
                    # Converter objetos para dict
                    json_data = [obj.__dict__ if hasattr(obj, '__dict__') else obj for obj in data]
                else:
                    json_data = data
                
                json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Backup salvo: {backup_file}")
            
            # Limpar backups antigos (manter apenas os 10 mais recentes)
            self._cleanup_old_backups(backup_name)
            
        except Exception as e:
            logger.error(f"Erro ao salvar backup {backup_name}: {e}")
    
    def _cleanup_old_backups(self, backup_name: str, keep_count: int = 10):
        """Remove backups antigos."""
        try:
            pattern = f"{backup_name}_*.json"
            backup_files = list(self.backup_dir.glob(pattern))
            
            if len(backup_files) > keep_count:
                # Ordenar por data de modificação
                backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Remover os mais antigos
                for old_file in backup_files[keep_count:]:
                    old_file.unlink()
                    logger.debug(f"Backup antigo removido: {old_file}")
                    
        except Exception as e:
            logger.error(f"Erro ao limpar backups antigos: {e}")
    
    def load_backup_data(self, backup_name: str) -> Optional[Any]:
        """Carrega dados de backup mais recente."""
        try:
            pattern = f"{backup_name}_*.json"
            backup_files = list(self.backup_dir.glob(pattern))
            
            if not backup_files:
                logger.warning(f"Nenhum backup encontrado para: {backup_name}")
                return None
            
            # Pegar o mais recente
            latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_backup, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Backup carregado: {latest_backup}")
            return data
            
        except Exception as e:
            logger.error(f"Erro ao carregar backup {backup_name}: {e}")
            return None
    
    def try_fallback_sources(self, *args, **kwargs) -> Optional[Any]:
        """Tenta fontes de fallback em ordem de prioridade."""
        for source in self.fallback_sources:
            try:
                logger.info(f"Tentando fonte de fallback: {source['name']}")
                result = source['func'](*args, **kwargs)
                if result:
                    logger.info(f"Fallback bem-sucedido: {source['name']}")
                    return result
            except Exception as e:
                logger.warning(f"Fallback {source['name']} falhou: {e}")
                continue
        
        logger.error("Todos os fallbacks falharam")
        return None

class ResilientAPIClient:
    """Cliente de API com tratamento robusto de erros."""
    
    def __init__(self, error_handler: ErrorHandler, fallback_manager: FallbackManager):
        self.error_handler = error_handler
        self.fallback_manager = fallback_manager
        self.session = requests.Session()
    
    def make_request_with_retry(self, 
                               method: str, 
                               url: str, 
                               retry_config: Optional[RetryConfig] = None,
                               **kwargs) -> Optional[requests.Response]:
        """Faz requisição com retry automático."""
        retry_config = retry_config or RetryConfig()
        retry_count = 0
        
        while retry_count <= retry_config.max_retries:
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
                
            except Exception as e:
                context = {
                    'method': method,
                    'url': url,
                    'retry_count': retry_count,
                    'kwargs': {k: v for k, v in kwargs.items() if k != 'data'}
                }
                
                error_info = self.error_handler.handle_error(e, context)
                error_info.retry_count = retry_count
                
                if not self.error_handler.should_retry(error_info, retry_config):
                    logger.error(f"Não será feita nova tentativa para {url}")
                    break
                
                retry_count += 1
                if retry_count <= retry_config.max_retries:
                    delay = self.error_handler.calculate_retry_delay(retry_count, retry_config)
                    logger.info(f"Tentativa {retry_count}/{retry_config.max_retries} em {delay:.2f}s")
                    time.sleep(delay)
        
        return None
    
    def get_with_fallback(self, url: str, fallback_key: str = None, **kwargs) -> Optional[Any]:
        """GET com fallback automático."""
        # Tentar requisição principal
        response = self.make_request_with_retry('GET', url, **kwargs)
        
        if response:
            try:
                data = response.json()
                # Salvar como backup
                if fallback_key:
                    self.fallback_manager.save_backup_data(data, fallback_key)
                return data
            except json.JSONDecodeError as e:
                self.error_handler.handle_error(e, {'url': url, 'response_text': response.text[:500]})
        
        # Tentar fallbacks
        if fallback_key:
            logger.info(f"Tentando fallbacks para {fallback_key}")
            return self.fallback_manager.try_fallback_sources(url, **kwargs)
        
        return None

# Instâncias globais
error_handler = ErrorHandler()
fallback_manager = FallbackManager()
resilient_client = ResilientAPIClient(error_handler, fallback_manager)

# Decorador para tratamento automático de erros
def handle_errors(fallback_value=None, log_errors=True):
    """Decorador para tratamento automático de erros."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    context = {
                        'function': func.__name__,
                        'args': str(args)[:200],
                        'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
                    }
                    error_handler.handle_error(e, context)
                
                return fallback_value
        return wrapper
    return decorator

# Funções de conveniência
def get_error_stats() -> Dict[str, Any]:
    """Obtém estatísticas de erros."""
    return error_handler.get_error_stats()

def clear_error_history():
    """Limpa histórico de erros."""
    error_handler.errors.clear()
    error_handler.error_counts.clear()
    error_handler.last_errors.clear()
    logger.info("Histórico de erros limpo")

def add_fallback_source(name: str, source_func: Callable, priority: int = 1):
    """Adiciona fonte de fallback global."""
    fallback_manager.add_fallback_source(name, source_func, priority)

def save_backup(data: Any, name: str):
    """Salva backup global."""
    fallback_manager.save_backup_data(data, name)

def load_backup(name: str) -> Optional[Any]:
    """Carrega backup global."""
    return fallback_manager.load_backup_data(name)