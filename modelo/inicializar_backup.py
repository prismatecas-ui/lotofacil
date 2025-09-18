#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Inicialização e Gerenciamento de Backup - Lotofácil
Script principal para configurar e gerenciar o sistema de backup e versionamento.

Autor: Sistema de Predição Lotofácil
Versão: 1.0.0
Data: 2024
"""

import os
import sys
import json
import argparse
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Thread, Event
import signal

# Importar o sistema de backup
try:
    from backup_versionamento import GerenciadorBackupVersionamento, criar_backup_automatico
except ImportError:
    print("Erro: Não foi possível importar o módulo backup_versionamento.py")
    print("Certifique-se de que o arquivo está no mesmo diretório.")
    sys.exit(1)

class GerenciadorSistemaBackup:
    """
    Gerenciador principal do sistema de backup e versionamento.
    Responsável por inicializar, configurar e monitorar o sistema.
    """
    
    def __init__(self, config_path: str = "config_backup.json"):
        self.config_path = config_path
        self.config = {}
        self.gerenciador_backup = None
        self.thread_scheduler = None
        self.stop_event = Event()
        self.logger = None
        
        # Configurar manipulador de sinais
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Manipula sinais de interrupção para parada graceful."""
        print(f"\nRecebido sinal {signum}. Parando sistema de backup...")
        self.parar_sistema()
    
    def carregar_configuracao(self) -> bool:
        """
        Carrega a configuração do sistema de backup.
        
        Returns:
            bool: True se carregou com sucesso, False caso contrário
        """
        try:
            if not os.path.exists(self.config_path):
                print(f"Arquivo de configuração não encontrado: {self.config_path}")
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            print(f"Configuração carregada: {self.config_path}")
            return True
            
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return False
    
    def configurar_logging(self):
        """Configura o sistema de logging."""
        try:
            # Criar diretório de logs se não existir
            log_dir = Path(self.config['diretorios']['logs'])
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configurar logging
            log_config = self.config['notificacoes']['canais']['log_arquivo']
            log_file = log_config['arquivo']
            
            # Configurar formatação
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Configurar handler de arquivo
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            
            # Configurar handler de console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Configurar logger principal
            self.logger = logging.getLogger('SistemaBackup')
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
            
            if self.config['notificacoes']['canais']['console']['habilitado']:
                self.logger.addHandler(console_handler)
            
            self.logger.info("Sistema de logging configurado")
            
        except Exception as e:
            print(f"Erro ao configurar logging: {e}")
    
    def verificar_dependencias(self) -> bool:
        """
        Verifica se todas as dependências estão instaladas.
        
        Returns:
            bool: True se todas as dependências estão OK
        """
        dependencias = [
            'numpy', 'pandas', 'scikit-learn', 'tensorflow',
            'xgboost', 'lightgbm', 'schedule'
        ]
        
        dependencias_faltando = []
        
        for dep in dependencias:
            try:
                __import__(dep)
            except ImportError:
                dependencias_faltando.append(dep)
        
        if dependencias_faltando:
            print("Dependências faltando:")
            for dep in dependencias_faltando:
                print(f"  - {dep}")
            print("\nInstale com: pip install " + " ".join(dependencias_faltando))
            return False
        
        return True
    
    def criar_estrutura_diretorios(self):
        """Cria a estrutura de diretórios necessária."""
        try:
            diretorios = self.config['diretorios']
            
            for nome, caminho in diretorios.items():
                Path(caminho).mkdir(parents=True, exist_ok=True)
                if self.logger:
                    self.logger.info(f"Diretório criado/verificado: {caminho}")
            
            print("Estrutura de diretórios criada com sucesso")
            
        except Exception as e:
            error_msg = f"Erro ao criar estrutura de diretórios: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
    
    def inicializar_gerenciador_backup(self) -> bool:
        """
        Inicializa o gerenciador de backup.
        
        Returns:
            bool: True se inicializou com sucesso
        """
        try:
            diretorio_base = self.config['diretorios']['base']
            self.gerenciador_backup = GerenciadorBackupVersionamento(diretorio_base)
            
            if self.logger:
                self.logger.info("Gerenciador de backup inicializado")
            
            return True
            
        except Exception as e:
            error_msg = f"Erro ao inicializar gerenciador de backup: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            return False
    
    def configurar_agendamentos(self):
        """Configura os agendamentos automáticos."""
        try:
            config_auto = self.config['backup_automatico']
            
            if not config_auto['habilitado']:
                if self.logger:
                    self.logger.info("Backup automático desabilitado")
                return
            
            # Agendar backups automáticos
            for horario in config_auto['horarios_execucao']:
                schedule.every().day.at(horario).do(self._executar_backup_automatico)
            
            # Agendar verificação de integridade
            manutencao = self.config['manutencao']
            if manutencao['verificacao_integridade']['habilitada']:
                dia_semana = manutencao['verificacao_integridade']['dia_semana']
                horario = manutencao['verificacao_integridade']['horario']
                
                if dia_semana == 7:  # Domingo
                    schedule.every().sunday.at(horario).do(self._verificar_integridade)
                elif dia_semana == 1:  # Segunda
                    schedule.every().monday.at(horario).do(self._verificar_integridade)
                # ... outros dias da semana
            
            # Agendar limpeza de temporários
            if manutencao['limpeza_temporarios']['habilitada']:
                horario = manutencao['limpeza_temporarios']['horario']
                schedule.every().day.at(horario).do(self._limpar_temporarios)
            
            if self.logger:
                self.logger.info("Agendamentos configurados")
            
        except Exception as e:
            error_msg = f"Erro ao configurar agendamentos: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
    
    def _executar_backup_automatico(self):
        """Executa backup automático agendado."""
        try:
            if self.logger:
                self.logger.info("Iniciando backup automático agendado")
            
            # Monitorar diretórios configurados
            diretorios = self.config['backup_automatico']['diretorios_monitorados']
            extensoes = self.config['backup_automatico']['extensoes_monitoradas']
            
            for diretorio in diretorios:
                if os.path.exists(diretorio):
                    self._backup_diretorio(diretorio, extensoes)
            
            if self.logger:
                self.logger.info("Backup automático concluído")
            
        except Exception as e:
            error_msg = f"Erro no backup automático: {e}"
            if self.logger:
                self.logger.error(error_msg)
    
    def _backup_diretorio(self, diretorio: str, extensoes: List[str]):
        """Faz backup de arquivos em um diretório específico."""
        try:
            for root, dirs, files in os.walk(diretorio):
                for file in files:
                    if any(file.endswith(ext) for ext in extensoes):
                        caminho_arquivo = os.path.join(root, file)
                        
                        # Verificar se deve ignorar o arquivo
                        ignorar = self.config['backup_automatico']['ignorar_arquivos']
                        if any(self._match_pattern(file, pattern) for pattern in ignorar):
                            continue
                        
                        # Criar backup do arquivo
                        self.gerenciador_backup.criar_backup(
                            caminho_arquivo,
                            tipo_mudanca="automatic",
                            descricao=f"Backup automático de {file}"
                        )
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Erro ao fazer backup do diretório {diretorio}: {e}")
    
    def _match_pattern(self, filename: str, pattern: str) -> bool:
        """Verifica se um nome de arquivo corresponde a um padrão."""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def _verificar_integridade(self):
        """Executa verificação de integridade agendada."""
        try:
            if self.logger:
                self.logger.info("Iniciando verificação de integridade")
            
            # Implementar verificação de integridade
            # Por enquanto, apenas log
            
            if self.logger:
                self.logger.info("Verificação de integridade concluída")
            
        except Exception as e:
            error_msg = f"Erro na verificação de integridade: {e}"
            if self.logger:
                self.logger.error(error_msg)
    
    def _limpar_temporarios(self):
        """Executa limpeza de arquivos temporários."""
        try:
            if self.logger:
                self.logger.info("Iniciando limpeza de temporários")
            
            # Implementar limpeza de temporários
            # Por enquanto, apenas log
            
            if self.logger:
                self.logger.info("Limpeza de temporários concluída")
            
        except Exception as e:
            error_msg = f"Erro na limpeza de temporários: {e}"
            if self.logger:
                self.logger.error(error_msg)
    
    def _executar_scheduler(self):
        """Thread para executar o scheduler."""
        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)  # Verificar a cada minuto
    
    def inicializar_sistema(self) -> bool:
        """
        Inicializa todo o sistema de backup.
        
        Returns:
            bool: True se inicializou com sucesso
        """
        print("=== Inicializando Sistema de Backup e Versionamento ===")
        
        # 1. Carregar configuração
        if not self.carregar_configuracao():
            return False
        
        # 2. Configurar logging
        self.configurar_logging()
        
        # 3. Verificar dependências
        if not self.verificar_dependencias():
            return False
        
        # 4. Criar estrutura de diretórios
        self.criar_estrutura_diretorios()
        
        # 5. Inicializar gerenciador de backup
        if not self.inicializar_gerenciador_backup():
            return False
        
        # 6. Configurar agendamentos
        self.configurar_agendamentos()
        
        # 7. Iniciar thread do scheduler
        if self.config['backup_automatico']['habilitado']:
            self.thread_scheduler = Thread(target=self._executar_scheduler, daemon=True)
            self.thread_scheduler.start()
        
        print("Sistema de backup inicializado com sucesso!")
        if self.logger:
            self.logger.info("Sistema de backup inicializado com sucesso")
        
        return True
    
    def parar_sistema(self):
        """Para o sistema de backup graciosamente."""
        print("Parando sistema de backup...")
        
        self.stop_event.set()
        
        if self.thread_scheduler and self.thread_scheduler.is_alive():
            self.thread_scheduler.join(timeout=5)
        
        if self.logger:
            self.logger.info("Sistema de backup parado")
        
        print("Sistema de backup parado com sucesso.")
    
    def status_sistema(self) -> Dict[str, Any]:
        """Retorna o status atual do sistema."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'configuracao_carregada': bool(self.config),
            'gerenciador_inicializado': self.gerenciador_backup is not None,
            'scheduler_ativo': self.thread_scheduler is not None and self.thread_scheduler.is_alive(),
            'backup_automatico_habilitado': self.config.get('backup_automatico', {}).get('habilitado', False),
            'proximos_agendamentos': []
        }
        
        # Adicionar próximos agendamentos
        try:
            jobs = schedule.jobs
            for job in jobs[:5]:  # Próximos 5 agendamentos
                status['proximos_agendamentos'].append({
                    'funcao': job.job_func.__name__,
                    'proximo_execucao': job.next_run.isoformat() if job.next_run else None
                })
        except:
            pass
        
        return status
    
    def executar_backup_manual(self, caminho_modelo: str, descricao: str = "") -> bool:
        """
        Executa um backup manual de um modelo específico.
        
        Args:
            caminho_modelo: Caminho para o arquivo do modelo
            descricao: Descrição do backup
        
        Returns:
            bool: True se o backup foi criado com sucesso
        """
        try:
            if not self.gerenciador_backup:
                print("Sistema não inicializado")
                return False
            
            if not os.path.exists(caminho_modelo):
                print(f"Arquivo não encontrado: {caminho_modelo}")
                return False
            
            versao = self.gerenciador_backup.criar_backup(
                caminho_modelo,
                tipo_mudanca="manual",
                descricao=descricao or f"Backup manual de {os.path.basename(caminho_modelo)}"
            )
            
            print(f"Backup criado com sucesso! Versão: {versao}")
            if self.logger:
                self.logger.info(f"Backup manual criado: {caminho_modelo} -> {versao}")
            
            return True
            
        except Exception as e:
            error_msg = f"Erro ao criar backup manual: {e}"
            print(error_msg)
            if self.logger:
                self.logger.error(error_msg)
            return False

def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(
        description="Sistema de Backup e Versionamento - Lotofácil"
    )
    
    parser.add_argument(
        'comando',
        choices=['iniciar', 'parar', 'status', 'backup', 'config'],
        help='Comando a executar'
    )
    
    parser.add_argument(
        '--config',
        default='config_backup.json',
        help='Caminho para o arquivo de configuração'
    )
    
    parser.add_argument(
        '--modelo',
        help='Caminho para o modelo (usado com comando backup)'
    )
    
    parser.add_argument(
        '--descricao',
        help='Descrição do backup (usado com comando backup)'
    )
    
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Executar em modo daemon (apenas com comando iniciar)'
    )
    
    args = parser.parse_args()
    
    # Criar gerenciador
    gerenciador = GerenciadorSistemaBackup(args.config)
    
    if args.comando == 'iniciar':
        if gerenciador.inicializar_sistema():
            if args.daemon:
                print("Sistema executando em modo daemon. Pressione Ctrl+C para parar.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    gerenciador.parar_sistema()
            else:
                print("Sistema inicializado. Use 'python inicializar_backup.py status' para verificar.")
        else:
            print("Falha ao inicializar sistema")
            sys.exit(1)
    
    elif args.comando == 'status':
        if gerenciador.carregar_configuracao():
            status = gerenciador.status_sistema()
            print("=== Status do Sistema de Backup ===")
            for chave, valor in status.items():
                print(f"{chave}: {valor}")
        else:
            print("Erro ao carregar configuração")
    
    elif args.comando == 'backup':
        if not args.modelo:
            print("Erro: --modelo é obrigatório para o comando backup")
            sys.exit(1)
        
        if gerenciador.inicializar_sistema():
            sucesso = gerenciador.executar_backup_manual(
                args.modelo,
                args.descricao or ""
            )
            if not sucesso:
                sys.exit(1)
        else:
            print("Falha ao inicializar sistema")
            sys.exit(1)
    
    elif args.comando == 'config':
        if os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(json.dumps(config, indent=2, ensure_ascii=False))
        else:
            print(f"Arquivo de configuração não encontrado: {args.config}")
    
    elif args.comando == 'parar':
        # Para implementar parada de processo em execução
        print("Comando 'parar' não implementado ainda")

if __name__ == "__main__":
    main()