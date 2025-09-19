#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Logs Estruturados para Monitoramento de Performance
Provê logging estruturado com contexto rico para análise e debugging
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import contextmanager
import threading
import time
import uuid

class LogLevel(Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """Categorias de log"""
    PREDICTION = "prediction"
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    API = "api"
    DATABASE = "database"
    ALERT = "alert"
    TRAINING = "training"
    ERROR = "error"
    AUDIT = "audit"

@dataclass
class LogContext:
    """Contexto do log"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Métricas de performance"""
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    database_queries: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None

@dataclass
class StructuredLogEntry:
    """Entrada de log estruturada"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    context: LogContext
    performance: Optional[PerformanceMetrics] = None
    exception: Optional[Dict[str, Any]] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        result = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'category': self.category.value,
            'message': self.message,
            'context': asdict(self.context),
        }
        
        if self.performance:
            result['performance'] = asdict(self.performance)
        
        if self.exception:
            result['exception'] = self.exception
        
        if self.extra_data:
            result['extra_data'] = self.extra_data
        
        return result
    
    def to_json(self) -> str:
        """Converte para JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))

class StructuredFormatter(logging.Formatter):
    """Formatter para logs estruturados"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Se o record já tem dados estruturados, usa eles
        if hasattr(record, 'structured_data'):
            return record.structured_data.to_json()
        
        # Caso contrário, cria entrada estruturada básica
        context = LogContext(
            component=record.name,
            metadata={'filename': record.filename, 'lineno': record.lineno}
        )
        
        level_map = {
            logging.DEBUG: LogLevel.DEBUG,
            logging.INFO: LogLevel.INFO,
            logging.WARNING: LogLevel.WARNING,
            logging.ERROR: LogLevel.ERROR,
            logging.CRITICAL: LogLevel.CRITICAL
        }
        
        entry = StructuredLogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=level_map.get(record.levelno, LogLevel.INFO),
            category=LogCategory.SYSTEM,
            message=record.getMessage(),
            context=context
        )
        
        # Adiciona informações de exceção se houver
        if record.exc_info:
            entry.exception = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return entry.to_json()

class StructuredLogger:
    """Logger estruturado principal"""
    
    def __init__(self, name: str = "lotofacil_ai", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Context local para thread
        self._local = threading.local()
        
        # Configura loggers
        self._setup_loggers()
        
        # Métricas de performance
        self.performance_tracker = PerformanceTracker()
    
    def _setup_loggers(self):
        """Configura os loggers"""
        # Logger principal
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Handler para arquivo geral
        general_handler = logging.FileHandler(
            self.log_dir / "application.jsonl",
            encoding='utf-8'
        )
        general_handler.setFormatter(StructuredFormatter())
        general_handler.setLevel(logging.INFO)
        self.logger.addHandler(general_handler)
        
        # Handler para erros
        error_handler = logging.FileHandler(
            self.log_dir / "errors.jsonl",
            encoding='utf-8'
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
        # Handler para performance
        performance_handler = logging.FileHandler(
            self.log_dir / "performance.jsonl",
            encoding='utf-8'
        )
        performance_handler.setFormatter(StructuredFormatter())
        performance_handler.setLevel(logging.INFO)
        
        # Logger específico para performance
        self.performance_logger = logging.getLogger(f"{self.name}.performance")
        self.performance_logger.addHandler(performance_handler)
        self.performance_logger.setLevel(logging.INFO)
        
        # Handler para console (desenvolvimento)
        if self._is_development():
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(StructuredFormatter())
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)
    
    def _is_development(self) -> bool:
        """Verifica se está em ambiente de desenvolvimento"""
        import os
        return os.getenv('ENVIRONMENT', 'development').lower() == 'development'
    
    def set_context(self, **kwargs):
        """Define contexto para a thread atual"""
        if not hasattr(self._local, 'context'):
            self._local.context = LogContext()
        
        for key, value in kwargs.items():
            if hasattr(self._local.context, key):
                setattr(self._local.context, key, value)
            else:
                self._local.context.metadata[key] = value
    
    def get_context(self) -> LogContext:
        """Obtém contexto da thread atual"""
        if not hasattr(self._local, 'context'):
            self._local.context = LogContext()
        return self._local.context
    
    def clear_context(self):
        """Limpa contexto da thread atual"""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager para contexto temporário"""
        old_context = self.get_context()
        
        # Salva valores antigos
        old_values = {}
        for key, value in kwargs.items():
            if hasattr(old_context, key):
                old_values[key] = getattr(old_context, key)
            else:
                old_values[key] = old_context.metadata.get(key)
        
        try:
            self.set_context(**kwargs)
            yield
        finally:
            # Restaura valores antigos
            for key, value in old_values.items():
                if hasattr(old_context, key):
                    setattr(old_context, key, value)
                else:
                    if value is not None:
                        old_context.metadata[key] = value
                    elif key in old_context.metadata:
                        del old_context.metadata[key]
    
    def _log(self, level: LogLevel, category: LogCategory, message: str, 
             performance: Optional[PerformanceMetrics] = None,
             exception: Optional[Exception] = None,
             **extra_data):
        """Método interno de log"""
        context = self.get_context()
        
        # Cria entrada estruturada
        entry = StructuredLogEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            context=context,
            performance=performance,
            extra_data=extra_data
        )
        
        # Adiciona informações de exceção
        if exception:
            entry.exception = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exception(type(exception), exception, exception.__traceback__)
            }
        
        # Cria record do logging
        record = logging.LogRecord(
            name=self.logger.name,
            level=getattr(logging, level.value),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.structured_data = entry
        
        # Envia para logger apropriado
        if category == LogCategory.PERFORMANCE:
            self.performance_logger.handle(record)
        else:
            self.logger.handle(record)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **extra_data):
        """Log de debug"""
        self._log(LogLevel.DEBUG, category, message, **extra_data)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **extra_data):
        """Log de informação"""
        self._log(LogLevel.INFO, category, message, **extra_data)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **extra_data):
        """Log de aviso"""
        self._log(LogLevel.WARNING, category, message, **extra_data)
    
    def error(self, message: str, category: LogCategory = LogCategory.ERROR, 
              exception: Optional[Exception] = None, **extra_data):
        """Log de erro"""
        self._log(LogLevel.ERROR, category, message, exception=exception, **extra_data)
    
    def critical(self, message: str, category: LogCategory = LogCategory.ERROR,
                 exception: Optional[Exception] = None, **extra_data):
        """Log crítico"""
        self._log(LogLevel.CRITICAL, category, message, exception=exception, **extra_data)
    
    def log_prediction(self, prediction_data: Dict[str, Any], confidence: float, 
                      execution_time_ms: float, **extra_data):
        """Log específico para predições"""
        performance = PerformanceMetrics(execution_time_ms=execution_time_ms)
        
        self._log(
            LogLevel.INFO,
            LogCategory.PREDICTION,
            f"Predição realizada com confiança {confidence:.2%}",
            performance=performance,
            prediction_data=prediction_data,
            confidence=confidence,
            **extra_data
        )
    
    def log_accuracy(self, accuracy: float, total_predictions: int, 
                    correct_predictions: int, **extra_data):
        """Log específico para acurácia"""
        self._log(
            LogLevel.INFO,
            LogCategory.ACCURACY,
            f"Acurácia calculada: {accuracy:.2%} ({correct_predictions}/{total_predictions})",
            accuracy=accuracy,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            **extra_data
        )
    
    def log_performance(self, operation: str, execution_time_ms: float,
                       memory_usage_mb: Optional[float] = None,
                       **extra_data):
        """Log específico para performance"""
        performance = PerformanceMetrics(
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb
        )
        
        self._log(
            LogLevel.INFO,
            LogCategory.PERFORMANCE,
            f"Operação '{operation}' executada em {execution_time_ms:.2f}ms",
            performance=performance,
            operation=operation,
            **extra_data
        )
    
    def log_api_request(self, method: str, endpoint: str, status_code: int,
                       execution_time_ms: float, **extra_data):
        """Log específico para requisições API"""
        performance = PerformanceMetrics(execution_time_ms=execution_time_ms)
        
        level = LogLevel.INFO if status_code < 400 else LogLevel.ERROR
        
        self._log(
            level,
            LogCategory.API,
            f"{method} {endpoint} - {status_code} ({execution_time_ms:.2f}ms)",
            performance=performance,
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            **extra_data
        )
    
    def log_database_operation(self, operation: str, table: str, 
                              execution_time_ms: float, rows_affected: int = 0,
                              **extra_data):
        """Log específico para operações de banco"""
        performance = PerformanceMetrics(
            execution_time_ms=execution_time_ms,
            database_queries=1
        )
        
        self._log(
            LogLevel.INFO,
            LogCategory.DATABASE,
            f"DB {operation} em {table}: {rows_affected} linhas ({execution_time_ms:.2f}ms)",
            performance=performance,
            operation=operation,
            table=table,
            rows_affected=rows_affected,
            **extra_data
        )
    
    def log_alert(self, alert_type: str, severity: str, message: str, **extra_data):
        """Log específico para alertas"""
        level_map = {
            'low': LogLevel.INFO,
            'medium': LogLevel.WARNING,
            'high': LogLevel.ERROR,
            'critical': LogLevel.CRITICAL
        }
        
        level = level_map.get(severity.lower(), LogLevel.WARNING)
        
        self._log(
            level,
            LogCategory.ALERT,
            f"Alerta {alert_type} ({severity}): {message}",
            alert_type=alert_type,
            severity=severity,
            **extra_data
        )
    
    @contextmanager
    def performance_timer(self, operation: str, category: LogCategory = LogCategory.PERFORMANCE):
        """Context manager para medir performance"""
        start_time = time.time()
        start_memory = self.performance_tracker.get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.performance_tracker.get_memory_usage()
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_diff = end_memory - start_memory if end_memory and start_memory else None
            
            performance = PerformanceMetrics(
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_diff
            )
            
            self._log(
                LogLevel.INFO,
                category,
                f"Operação '{operation}' concluída",
                performance=performance,
                operation=operation
            )

class PerformanceTracker:
    """Rastreador de métricas de performance"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def get_memory_usage(self) -> Optional[float]:
        """Obtém uso atual de memória em MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
    
    def get_cpu_usage(self) -> Optional[float]:
        """Obtém uso atual de CPU"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return None

class LogAnalyzer:
    """Analisador de logs estruturados"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
    
    def analyze_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analisa logs de performance"""
        performance_file = self.log_dir / "performance.jsonl"
        
        if not performance_file.exists():
            return {'error': 'Arquivo de performance não encontrado'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        operations = {}
        
        try:
            with open(performance_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(entry['timestamp'])
                        
                        if timestamp < cutoff_time:
                            continue
                        
                        operation = entry.get('extra_data', {}).get('operation', 'unknown')
                        performance = entry.get('performance', {})
                        execution_time = performance.get('execution_time_ms')
                        
                        if execution_time is not None:
                            if operation not in operations:
                                operations[operation] = []
                            operations[operation].append(execution_time)
                    
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            # Calcula estatísticas
            stats = {}
            for operation, times in operations.items():
                if times:
                    stats[operation] = {
                        'count': len(times),
                        'avg_ms': sum(times) / len(times),
                        'min_ms': min(times),
                        'max_ms': max(times),
                        'total_ms': sum(times)
                    }
            
            return {
                'period_hours': hours,
                'operations': stats,
                'total_operations': sum(len(times) for times in operations.values())
            }
            
        except Exception as e:
            return {'error': f'Erro ao analisar logs: {e}'}
    
    def analyze_errors(self, hours: int = 24) -> Dict[str, Any]:
        """Analisa logs de erro"""
        error_file = self.log_dir / "errors.jsonl"
        
        if not error_file.exists():
            return {'error': 'Arquivo de erros não encontrado'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        errors = []
        error_types = {}
        
        try:
            with open(error_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(entry['timestamp'])
                        
                        if timestamp < cutoff_time:
                            continue
                        
                        errors.append(entry)
                        
                        # Conta tipos de erro
                        exception = entry.get('exception', {})
                        error_type = exception.get('type', 'Unknown')
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            return {
                'period_hours': hours,
                'total_errors': len(errors),
                'error_types': error_types,
                'recent_errors': errors[-10:]  # Últimos 10 erros
            }
            
        except Exception as e:
            return {'error': f'Erro ao analisar logs de erro: {e}'}

# Instância global do logger estruturado
structured_logger = StructuredLogger()

# Funções de conveniência
def set_log_context(**kwargs):
    """Define contexto de log"""
    structured_logger.set_context(**kwargs)

def clear_log_context():
    """Limpa contexto de log"""
    structured_logger.clear_context()

def log_context(**kwargs):
    """Context manager para contexto de log"""
    return structured_logger.context(**kwargs)

def log_prediction(prediction_data: Dict[str, Any], confidence: float, 
                  execution_time_ms: float, **extra_data):
    """Log de predição"""
    structured_logger.log_prediction(prediction_data, confidence, execution_time_ms, **extra_data)

def log_accuracy(accuracy: float, total_predictions: int, 
                correct_predictions: int, **extra_data):
    """Log de acurácia"""
    structured_logger.log_accuracy(accuracy, total_predictions, correct_predictions, **extra_data)

def log_performance(operation: str, execution_time_ms: float, **extra_data):
    """Log de performance"""
    structured_logger.log_performance(operation, execution_time_ms, **extra_data)

def log_api_request(method: str, endpoint: str, status_code: int,
                   execution_time_ms: float, **extra_data):
    """Log de requisição API"""
    structured_logger.log_api_request(method, endpoint, status_code, execution_time_ms, **extra_data)

def log_alert(alert_type: str, severity: str, message: str, **extra_data):
    """Log de alerta"""
    structured_logger.log_alert(alert_type, severity, message, **extra_data)

def performance_timer(operation: str):
    """Timer de performance"""
    return structured_logger.performance_timer(operation)

if __name__ == "__main__":
    # Teste do sistema de logs
    print("Testando sistema de logs estruturados...")
    
    # Configura contexto
    set_log_context(request_id="test-123", user_id="user-456")
    
    # Testa diferentes tipos de log
    structured_logger.info("Sistema iniciado", category=LogCategory.SYSTEM)
    
    # Testa log de predição
    log_prediction(
        prediction_data={'numbers': [1, 2, 3, 4, 5]},
        confidence=0.85,
        execution_time_ms=150.5
    )
    
    # Testa log de performance
    with performance_timer("test_operation"):
        time.sleep(0.1)
    
    # Testa log de erro
    try:
        raise ValueError("Erro de teste")
    except Exception as e:
        structured_logger.error("Erro durante teste", exception=e)
    
    print("Teste concluído. Verifique os arquivos de log em 'logs/'")