"""Sistema de cache avançado para otimização de requisições da API Caixa.

Este módulo implementa diferentes estratégias de cache para melhorar
a performance e reduzir a carga na API da Caixa.
"""

import json
import time
import hashlib
import pickle
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import logging
from abc import ABC, abstractmethod
# Evitar importação circular - ConcursoResult será importado quando necessário

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configurações do sistema de cache."""
    cache_dir: str = "cache"
    default_ttl: int = 3600  # 1 hora em segundos
    max_memory_items: int = 1000
    enable_disk_cache: bool = True
    enable_memory_cache: bool = True
    enable_database_cache: bool = True
    compression_enabled: bool = True
    auto_cleanup: bool = True
    cleanup_interval: int = 86400  # 24 horas

@dataclass
class CacheEntry:
    """Entrada do cache com metadados."""
    key: str
    value: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Verifica se a entrada expirou."""
        return time.time() - self.timestamp > self.ttl
    
    def update_access(self):
        """Atualiza estatísticas de acesso."""
        self.access_count += 1
        self.last_access = time.time()

class CacheStrategy(ABC):
    """Interface para estratégias de cache."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Recupera valor do cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Armazena valor no cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove valor do cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Limpa todo o cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        pass

class MemoryCache(CacheStrategy):
    """Cache em memória com LRU."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _cleanup_expired(self):
        """Remove entradas expiradas."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.debug(f"Removidas {len(expired_keys)} entradas expiradas do cache de memória")
    
    def _evict_lru(self):
        """Remove entrada menos recentemente usada."""
        if not self.cache:
            return
        
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_access)
        del self.cache[lru_key]
        logger.debug(f"Entrada LRU removida: {lru_key}")
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.update_access()
                    self.hits += 1
                    return entry.value
                else:
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        with self.lock:
            try:
                ttl = ttl or self.config.default_ttl
                
                # Calcular tamanho aproximado
                size_bytes = len(pickle.dumps(value))
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl,
                    size_bytes=size_bytes
                )
                entry.update_access()
                
                # Verificar limite de itens
                if len(self.cache) >= self.config.max_memory_items:
                    self._evict_lru()
                
                self.cache[key] = entry
                return True
                
            except Exception as e:
                logger.error(f"Erro ao armazenar no cache de memória: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> bool:
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                'type': 'memory',
                'items': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'total_size_bytes': total_size,
                'max_items': self.config.max_memory_items
            }

class DiskCache(CacheStrategy):
    """Cache em disco com compressão opcional."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _get_file_path(self, key: str) -> Path:
        """Gera caminho do arquivo para a chave."""
        # Usar hash para evitar problemas com caracteres especiais
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serializa entrada para armazenamento."""
        data = pickle.dumps(entry)
        if self.config.compression_enabled:
            import gzip
            data = gzip.compress(data)
        return data
    
    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserializa entrada do armazenamento."""
        if self.config.compression_enabled:
            import gzip
            data = gzip.decompress(data)
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                self.misses += 1
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    entry = self._deserialize_entry(f.read())
                
                if entry.is_expired():
                    file_path.unlink(missing_ok=True)
                    self.misses += 1
                    return None
                
                entry.update_access()
                
                # Atualizar arquivo com nova estatística de acesso
                with open(file_path, 'wb') as f:
                    f.write(self._serialize_entry(entry))
                
                self.hits += 1
                return entry.value
                
            except Exception as e:
                logger.error(f"Erro ao ler cache de disco: {e}")
                file_path.unlink(missing_ok=True)
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        with self.lock:
            try:
                ttl = ttl or self.config.default_ttl
                file_path = self._get_file_path(key)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl
                )
                entry.update_access()
                
                with open(file_path, 'wb') as f:
                    data = self._serialize_entry(entry)
                    f.write(data)
                    entry.size_bytes = len(data)
                
                return True
                
            except Exception as e:
                logger.error(f"Erro ao armazenar no cache de disco: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self.lock:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
    
    def clear(self) -> bool:
        with self.lock:
            try:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
                self.hits = 0
                self.misses = 0
                return True
            except Exception as e:
                logger.error(f"Erro ao limpar cache de disco: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'type': 'disk',
                'items': len(cache_files),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'total_size_bytes': total_size,
                'cache_dir': str(self.cache_dir)
            }

class DatabaseCache(CacheStrategy):
    """Cache em banco SQLite."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.db_path = Path(config.cache_dir) / "cache.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self._init_database()
    
    def _init_database(self):
        """Inicializa o banco de dados."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    ttl INTEGER,
                    access_count INTEGER DEFAULT 0,
                    last_access REAL,
                    size_bytes INTEGER DEFAULT 0
                )
            """)
            
            # Índices para performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_access ON cache_entries(last_access)")
            conn.commit()
    
    def _cleanup_expired(self):
        """Remove entradas expiradas do banco."""
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE (? - timestamp) > ttl",
                (current_time,)
            )
            deleted_count = cursor.rowcount
            conn.commit()
        
        if deleted_count > 0:
            logger.debug(f"Removidas {deleted_count} entradas expiradas do cache de banco")
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT value, timestamp, ttl FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        self.misses += 1
                        return None
                    
                    value_blob, timestamp, ttl = row
                    
                    # Verificar expiração
                    if time.time() - timestamp > ttl:
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        self.misses += 1
                        return None
                    
                    # Atualizar estatísticas de acesso
                    conn.execute(
                        "UPDATE cache_entries SET access_count = access_count + 1, last_access = ? WHERE key = ?",
                        (time.time(), key)
                    )
                    conn.commit()
                    
                    self.hits += 1
                    return pickle.loads(value_blob)
                    
            except Exception as e:
                logger.error(f"Erro ao ler cache de banco: {e}")
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        with self.lock:
            try:
                ttl = ttl or self.config.default_ttl
                value_blob = pickle.dumps(value)
                current_time = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, timestamp, ttl, access_count, last_access, size_bytes)
                        VALUES (?, ?, ?, ?, 1, ?, ?)
                        """,
                        (key, value_blob, current_time, ttl, current_time, len(value_blob))
                    )
                    conn.commit()
                
                return True
                
            except Exception as e:
                logger.error(f"Erro ao armazenar no cache de banco: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Erro ao deletar do cache de banco: {e}")
                return False
    
    def clear(self) -> bool:
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries")
                    conn.commit()
                self.hits = 0
                self.misses = 0
                return True
            except Exception as e:
                logger.error(f"Erro ao limpar cache de banco: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache_entries")
                    count, total_size = cursor.fetchone()
                    
                    total_requests = self.hits + self.misses
                    hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
                    
                    return {
                        'type': 'database',
                        'items': count or 0,
                        'hits': self.hits,
                        'misses': self.misses,
                        'hit_rate': round(hit_rate, 2),
                        'total_size_bytes': total_size or 0,
                        'db_path': str(self.db_path)
                    }
            except Exception as e:
                logger.error(f"Erro ao obter estatísticas do cache de banco: {e}")
                return {'type': 'database', 'error': str(e)}

class MultiLevelCache:
    """Cache multi-nível combinando memória, disco e banco."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.strategies: List[CacheStrategy] = []
        
        # Inicializar estratégias habilitadas
        if self.config.enable_memory_cache:
            self.strategies.append(MemoryCache(self.config))
        
        if self.config.enable_disk_cache:
            self.strategies.append(DiskCache(self.config))
        
        if self.config.enable_database_cache:
            self.strategies.append(DatabaseCache(self.config))
        
        self.lock = threading.RLock()
        
        # Iniciar limpeza automática se habilitada
        if self.config.auto_cleanup:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Inicia thread de limpeza automática."""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval)
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Erro na limpeza automática: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("Thread de limpeza automática iniciada")
    
    def _cleanup_expired(self):
        """Executa limpeza em todas as estratégias."""
        for strategy in self.strategies:
            if hasattr(strategy, '_cleanup_expired'):
                try:
                    strategy._cleanup_expired()
                except Exception as e:
                    logger.error(f"Erro na limpeza de {type(strategy).__name__}: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Busca valor em todas as estratégias (L1 -> L2 -> L3)."""
        with self.lock:
            for i, strategy in enumerate(self.strategies):
                value = strategy.get(key)
                if value is not None:
                    # Promover para níveis superiores (write-back)
                    for j in range(i):
                        self.strategies[j].set(key, value)
                    return value
            
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Armazena valor em todas as estratégias."""
        with self.lock:
            success = True
            for strategy in self.strategies:
                if not strategy.set(key, value, ttl):
                    success = False
            return success
    
    def delete(self, key: str) -> bool:
        """Remove valor de todas as estratégias."""
        with self.lock:
            success = True
            for strategy in self.strategies:
                if not strategy.delete(key):
                    success = False
            return success
    
    def clear(self) -> bool:
        """Limpa todas as estratégias."""
        with self.lock:
            success = True
            for strategy in self.strategies:
                if not strategy.clear():
                    success = False
            return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de todas as estratégias."""
        with self.lock:
            stats = {
                'config': asdict(self.config),
                'strategies': [strategy.get_stats() for strategy in self.strategies],
                'total_strategies': len(self.strategies)
            }
            
            # Calcular estatísticas agregadas
            total_hits = sum(s.get('hits', 0) for s in stats['strategies'])
            total_misses = sum(s.get('misses', 0) for s in stats['strategies'])
            total_requests = total_hits + total_misses
            
            stats['aggregate'] = {
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_requests': total_requests,
                'overall_hit_rate': round((total_hits / total_requests * 100) if total_requests > 0 else 0, 2)
            }
            
            return stats

# Instância global do cache
cache_service = MultiLevelCache()

# Funções de conveniência
def get_cached(key: str) -> Optional[Any]:
    """Função de conveniência para buscar no cache."""
    return cache_service.get(key)

def set_cached(key: str, value: Any, ttl: int = None) -> bool:
    """Função de conveniência para armazenar no cache."""
    return cache_service.set(key, value, ttl)

def delete_cached(key: str) -> bool:
    """Função de conveniência para remover do cache."""
    return cache_service.delete(key)

def clear_cache() -> bool:
    """Função de conveniência para limpar cache."""
    return cache_service.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Função de conveniência para obter estatísticas."""
    return cache_service.get_stats()

def cached_function(ttl: int = None, key_func: Callable = None):
    """Decorador para cache de funções."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Gerar chave do cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Tentar buscar no cache
            cached_result = get_cached(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Executar função e armazenar resultado
            result = func(*args, **kwargs)
            set_cached(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator