#!/usr/bin/env python3
"""
Script de inicialização da API de Predições Lotofácil

Este script configura e inicia a API com todas as dependências necessárias.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import time
import signal
import threading
from datetime import datetime

# Adicionar diretório pai ao path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from api_predicoes import APIPredicoes, criar_api
except ImportError as e:
    print(f"Erro ao importar API: {e}")
    print("Certifique-se de que todos os módulos estão instalados")
    sys.exit(1)

class GerenciadorAPI:
    """
    Gerenciador principal da API com configurações avançadas
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "./api/config_api.json"
        self.config = self._carregar_configuracao()
        self.api = None
        self.running = False
        self._setup_logging()
        
    def _carregar_configuracao(self) -> Dict[str, Any]:
        """
        Carrega configuração da API
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"Configuração carregada de: {self.config_path}")
                return config
            else:
                print(f"Arquivo de configuração não encontrado: {self.config_path}")
                return self._configuracao_padrao()
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return self._configuracao_padrao()
    
    def _configuracao_padrao(self) -> Dict[str, Any]:
        """
        Configuração padrão caso não exista arquivo
        """
        return {
            "api": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False,
                "threaded": True
            },
            "cache": {
                "max_size": 1000,
                "ttl_seconds": 300,
                "enabled": True
            },
            "logging": {
                "level": "INFO",
                "file": "api_predicoes.log"
            }
        }
    
    def _setup_logging(self):
        """
        Configura sistema de logging
        """
        log_config = self.config.get('logging', {})
        
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_file = log_config.get('file', 'api_predicoes.log')
        
        # Configurar formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para arquivo
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        
        # Configurar logger raiz
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def verificar_dependencias(self) -> bool:
        """
        Verifica se todas as dependências estão disponíveis
        """
        dependencias = [
            'flask',
            'flask_cors',
            'numpy',
            'pandas',
            'tensorflow',
            'scikit-learn'
        ]
        
        faltando = []
        
        for dep in dependencias:
            try:
                __import__(dep.replace('-', '_'))
            except ImportError:
                faltando.append(dep)
        
        if faltando:
            self.logger.error(f"Dependências faltando: {', '.join(faltando)}")
            self.logger.info("Execute: pip install -r requirements_api.txt")
            return False
        
        self.logger.info("Todas as dependências estão disponíveis")
        return True
    
    def verificar_modelos(self) -> bool:
        """
        Verifica disponibilidade dos modelos
        """
        try:
            # Verificar se os módulos de modelo existem
            modulos_modelo = [
                'modelo.modelo_tensorflow2',
                'modelo.algoritmos_avancados',
                'modelo.analise_padroes'
            ]
            
            for modulo in modulos_modelo:
                try:
                    __import__(modulo)
                    self.logger.info(f"Módulo {modulo} disponível")
                except ImportError as e:
                    self.logger.warning(f"Módulo {modulo} não disponível: {e}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar modelos: {e}")
            return False
    
    def verificar_dados(self) -> bool:
        """
        Verifica disponibilidade dos dados
        """
        try:
            dados_config = self.config.get('dados', {})
            fonte_principal = dados_config.get('fonte_principal', './base/dados_lotofacil.xlsx')
            
            if os.path.exists(fonte_principal):
                self.logger.info(f"Dados encontrados: {fonte_principal}")
                return True
            else:
                self.logger.warning(f"Dados não encontrados: {fonte_principal}")
                
                # Verificar fonte backup
                fonte_backup = dados_config.get('fonte_backup')
                if fonte_backup and os.path.exists(fonte_backup):
                    self.logger.info(f"Usando dados backup: {fonte_backup}")
                    return True
                
                self.logger.warning("Nenhuma fonte de dados encontrada")
                return False
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar dados: {e}")
            return False
    
    def pre_inicializacao(self) -> bool:
        """
        Executa verificações antes de iniciar a API
        """
        self.logger.info("Iniciando verificações pré-inicialização...")
        
        # Verificar dependências
        if not self.verificar_dependencias():
            return False
        
        # Verificar modelos
        if not self.verificar_modelos():
            self.logger.warning("Alguns modelos podem não funcionar corretamente")
        
        # Verificar dados
        if not self.verificar_dados():
            self.logger.warning("API funcionará com dados limitados")
        
        self.logger.info("Verificações pré-inicialização concluídas")
        return True
    
    def iniciar_api(self):
        """
        Inicia a API
        """
        try:
            # Verificações pré-inicialização
            if not self.pre_inicializacao():
                self.logger.error("Falha nas verificações pré-inicialização")
                return False
            
            # Criar instância da API
            self.logger.info("Criando instância da API...")
            self.api = criar_api()
            
            # Configurar parâmetros
            api_config = self.config.get('api', {})
            host = api_config.get('host', '0.0.0.0')
            port = api_config.get('port', 5000)
            debug = api_config.get('debug', False)
            
            self.logger.info(f"Iniciando API em {host}:{port}")
            self.logger.info(f"Debug mode: {debug}")
            
            # Configurar handlers de sinal
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.running = True
            
            # Iniciar API
            self.api.executar(host=host, port=port, debug=debug)
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar API: {e}")
            return False
    
    def _signal_handler(self, signum, frame):
        """
        Handler para sinais de interrupção
        """
        self.logger.info(f"Sinal recebido: {signum}")
        self.parar_api()
    
    def parar_api(self):
        """
        Para a API graciosamente
        """
        self.logger.info("Parando API...")
        self.running = False
        
        if self.api:
            # Limpar cache
            if hasattr(self.api, 'cache'):
                self.api.cache.clear()
            
            self.logger.info("API parada com sucesso")
    
    def status_api(self) -> Dict[str, Any]:
        """
        Retorna status atual da API
        """
        return {
            'running': self.running,
            'config_loaded': bool(self.config),
            'api_instance': self.api is not None,
            'timestamp': datetime.now().isoformat()
        }
    
    def gerar_relatorio_startup(self):
        """
        Gera relatório de inicialização
        """
        relatorio = []
        relatorio.append("=" * 60)
        relatorio.append("API PREDIÇÕES LOTOFÁCIL - RELATÓRIO DE INICIALIZAÇÃO")
        relatorio.append("=" * 60)
        relatorio.append(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        relatorio.append(f"Configuração: {self.config_path}")
        relatorio.append("")
        
        # Informações da configuração
        api_config = self.config.get('api', {})
        relatorio.append("CONFIGURAÇÃO DA API:")
        relatorio.append(f"  Host: {api_config.get('host', 'N/A')}")
        relatorio.append(f"  Porta: {api_config.get('port', 'N/A')}")
        relatorio.append(f"  Debug: {api_config.get('debug', 'N/A')}")
        relatorio.append("")
        
        # Informações do cache
        cache_config = self.config.get('cache', {})
        relatorio.append("CONFIGURAÇÃO DO CACHE:")
        relatorio.append(f"  Habilitado: {cache_config.get('enabled', 'N/A')}")
        relatorio.append(f"  Tamanho máximo: {cache_config.get('max_size', 'N/A')}")
        relatorio.append(f"  TTL (segundos): {cache_config.get('ttl_seconds', 'N/A')}")
        relatorio.append("")
        
        # Modelos configurados
        modelos_config = self.config.get('modelos', {})
        relatorio.append("MODELOS CONFIGURADOS:")
        for nome, config in modelos_config.items():
            relatorio.append(f"  {nome}:")
            relatorio.append(f"    Tipo: {config.get('tipo', 'N/A')}")
            relatorio.append(f"    Ativo: {config.get('ativo', 'N/A')}")
            relatorio.append(f"    Preload: {config.get('preload', 'N/A')}")
        
        relatorio.append("")
        relatorio.append("=" * 60)
        
        relatorio_texto = "\n".join(relatorio)
        
        # Salvar relatório
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        arquivo_relatorio = f"relatorio_startup_{timestamp}.txt"
        
        try:
            with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
                f.write(relatorio_texto)
            self.logger.info(f"Relatório de inicialização salvo: {arquivo_relatorio}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar relatório: {e}")
        
        # Exibir no console
        print(relatorio_texto)


def main():
    """
    Função principal
    """
    parser = argparse.ArgumentParser(
        description='Inicializador da API de Predições Lotofácil',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python iniciar_api.py                    # Usar configuração padrão
  python iniciar_api.py --config custom.json  # Usar configuração personalizada
  python iniciar_api.py --host 127.0.0.1 --port 8080  # Sobrescrever host/porta
  python iniciar_api.py --debug             # Ativar modo debug
        """
    )
    
    parser.add_argument(
        '--config', 
        default='./api/config_api.json',
        help='Caminho para arquivo de configuração (padrão: ./api/config_api.json)'
    )
    
    parser.add_argument(
        '--host',
        help='Host da API (sobrescreve configuração)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Porta da API (sobrescreve configuração)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Ativar modo debug (sobrescreve configuração)'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Não gerar relatório de inicialização'
    )
    
    args = parser.parse_args()
    
    try:
        # Criar gerenciador
        gerenciador = GerenciadorAPI(args.config)
        
        # Sobrescrever configurações se fornecidas
        if args.host:
            gerenciador.config['api']['host'] = args.host
        
        if args.port:
            gerenciador.config['api']['port'] = args.port
        
        if args.debug:
            gerenciador.config['api']['debug'] = True
        
        # Gerar relatório de inicialização
        if not args.no_report:
            gerenciador.gerar_relatorio_startup()
        
        # Iniciar API
        print("\nIniciando API de Predições Lotofácil...")
        print("Pressione Ctrl+C para parar\n")
        
        gerenciador.iniciar_api()
        
    except KeyboardInterrupt:
        print("\nAPI interrompida pelo usuário")
    except Exception as e:
        print(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()