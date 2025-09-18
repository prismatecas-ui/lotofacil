"""Configurações para integração com a API da Caixa.

Este módulo contém as configurações de autenticação, headers,
e parâmetros necessários para a comunicação com a API oficial.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configurações da API da Caixa."""
    base_url: str
    timeout: int
    max_retries: int
    retry_delay: float
    cache_timeout: int
    rate_limit_delay: float
    user_agent: str
    headers: Dict[str, str]

class CaixaAPIConfig:
    """Gerenciador de configurações da API da Caixa."""
    
    def __init__(self):
        self._config = self._load_config()
    
    def _load_config(self) -> APIConfig:
        """Carrega as configurações da API."""
        return APIConfig(
            base_url=os.getenv(
                'CAIXA_API_BASE_URL', 
                'https://servicebus2.caixa.gov.br/portaldeloterias/api'
            ),
            timeout=int(os.getenv('CAIXA_API_TIMEOUT', '30')),
            max_retries=int(os.getenv('CAIXA_API_MAX_RETRIES', '3')),
            retry_delay=float(os.getenv('CAIXA_API_RETRY_DELAY', '1.0')),
            cache_timeout=int(os.getenv('CAIXA_API_CACHE_TIMEOUT', '300')),
            rate_limit_delay=float(os.getenv('CAIXA_API_RATE_LIMIT_DELAY', '0.1')),
            user_agent=os.getenv(
                'CAIXA_API_USER_AGENT',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ),
            headers=self._get_default_headers()
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Retorna os headers padrão para requisições."""
        return {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://loterias.caixa.gov.br/',
            'Origin': 'https://loterias.caixa.gov.br',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    
    @property
    def config(self) -> APIConfig:
        """Retorna a configuração atual."""
        return self._config
    
    def get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Retorna headers completos para requisições."""
        headers = {
            'User-Agent': self._config.user_agent,
            **self._config.headers
        }
        
        if additional_headers:
            headers.update(additional_headers)
        
        return headers
    
    def get_session_config(self) -> Dict[str, any]:
        """Retorna configurações para sessão requests."""
        return {
            'timeout': self._config.timeout,
            'headers': self.get_headers(),
            'verify': True,  # Verificar certificados SSL
            'allow_redirects': True,
            'stream': False
        }
    
    def update_config(self, **kwargs):
        """Atualiza configurações específicas."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Configuração {key} atualizada para: {value}")
            else:
                logger.warning(f"Configuração {key} não reconhecida")
    
    def reload_config(self):
        """Recarrega as configurações do ambiente."""
        self._config = self._load_config()
        logger.info("Configurações recarregadas")
    
    def validate_config(self) -> bool:
        """Valida se as configurações estão corretas."""
        try:
            # Validar URL base
            if not self._config.base_url.startswith(('http://', 'https://')):
                logger.error("URL base inválida")
                return False
            
            # Validar timeout
            if self._config.timeout <= 0:
                logger.error("Timeout deve ser maior que zero")
                return False
            
            # Validar max_retries
            if self._config.max_retries < 0:
                logger.error("Max retries deve ser maior ou igual a zero")
                return False
            
            # Validar delays
            if self._config.retry_delay < 0 or self._config.rate_limit_delay < 0:
                logger.error("Delays devem ser maiores ou iguais a zero")
                return False
            
            # Validar cache timeout
            if self._config.cache_timeout < 0:
                logger.error("Cache timeout deve ser maior ou igual a zero")
                return False
            
            logger.info("Configurações validadas com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação das configurações: {e}")
            return False
    
    def get_endpoint_url(self, endpoint: str) -> str:
        """Constrói URL completa para um endpoint."""
        base_url = self._config.base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        return f"{base_url}/{endpoint}"
    
    def __str__(self) -> str:
        """Representação string das configurações."""
        return f"CaixaAPIConfig(base_url={self._config.base_url}, timeout={self._config.timeout})"

# Instância global de configuração
api_config = CaixaAPIConfig()

# Funções de conveniência
def get_config() -> APIConfig:
    """Retorna a configuração atual."""
    return api_config.config

def get_headers(additional: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Retorna headers para requisições."""
    return api_config.get_headers(additional)

def get_endpoint_url(endpoint: str) -> str:
    """Constrói URL para endpoint."""
    return api_config.get_endpoint_url(endpoint)

def validate_api_config() -> bool:
    """Valida configurações da API."""
    return api_config.validate_config()