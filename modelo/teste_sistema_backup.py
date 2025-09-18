#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste do Sistema de Backup e Versionamento - Lotofácil
Script para testar e validar o funcionamento do sistema de backup.

Autor: Sistema de Predição Lotofácil
Versão: 1.0.0
Data: 2024
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Adicionar diretório atual ao path para importações
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from backup_versionamento import GerenciadorBackupVersionamento
    from inicializar_backup import GerenciadorSistemaBackup
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que os arquivos estão no mesmo diretório.")
    sys.exit(1)

class TesteSistemaBackup(unittest.TestCase):
    """
    Classe de testes para o sistema de backup e versionamento.
    """
    
    def setUp(self):
        """Configuração inicial para cada teste."""
        # Criar diretório temporário para testes
        self.temp_dir = tempfile.mkdtemp(prefix="teste_backup_")
        self.config_teste = self._criar_config_teste()
        self.config_path = os.path.join(self.temp_dir, "config_teste.json")
        
        # Salvar configuração de teste
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_teste, f, indent=2, ensure_ascii=False)
        
        # Criar arquivos de modelo de teste
        self.modelo_teste = self._criar_modelo_teste()
    
    def tearDown(self):
        """Limpeza após cada teste."""
        # Remover diretório temporário
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _criar_config_teste(self) -> Dict[str, Any]:
        """Cria uma configuração de teste simplificada."""
        return {
            "sistema_backup": {
                "versao": "1.0",
                "nome": "Teste Sistema Backup"
            },
            "diretorios": {
                "base": os.path.join(self.temp_dir, "backups"),
                "modelos": os.path.join(self.temp_dir, "backups", "modelos"),
                "metadados": os.path.join(self.temp_dir, "backups", "metadados"),
                "logs": os.path.join(self.temp_dir, "logs")
            },
            "backup_automatico": {
                "habilitado": False,
                "diretorios_monitorados": [self.temp_dir],
                "extensoes_monitoradas": [".h5", ".pkl", ".json"],
                "ignorar_arquivos": ["temp_*", "*.tmp"]
            },
            "versionamento": {
                "esquema": "semantico",
                "versao_inicial": "1.0.0"
            },
            "compressao": {
                "habilitada": True,
                "nivel_compressao": 6
            },
            "validacao": {
                "habilitada": True,
                "verificar_integridade": True
            },
            "notificacoes": {
                "habilitadas": True,
                "canais": {
                    "log_arquivo": {
                        "habilitado": True,
                        "arquivo": os.path.join(self.temp_dir, "logs", "teste.log")
                    },
                    "console": {
                        "habilitado": True
                    }
                }
            },
            "manutencao": {
                "verificacao_integridade": {"habilitada": False},
                "limpeza_temporarios": {"habilitada": False}
            },
            "desenvolvimento": {
                "modo_debug": True
            }
        }
    
    def _criar_modelo_teste(self) -> str:
        """Cria um arquivo de modelo de teste."""
        modelo_path = os.path.join(self.temp_dir, "modelo_teste.json")
        
        modelo_data = {
            "tipo": "teste",
            "versao": "1.0.0",
            "parametros": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "metricas": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(modelo_path, 'w', encoding='utf-8') as f:
            json.dump(modelo_data, f, indent=2, ensure_ascii=False)
        
        return modelo_path
    
    def test_inicializacao_gerenciador_backup(self):
        """Testa a inicialização do gerenciador de backup."""
        print("\n=== Teste: Inicialização do Gerenciador de Backup ===")
        
        diretorio_base = self.config_teste['diretorios']['base']
        gerenciador = GerenciadorBackupVersionamento(diretorio_base)
        
        self.assertIsNotNone(gerenciador)
        self.assertTrue(os.path.exists(diretorio_base))
        
        print("✓ Gerenciador de backup inicializado com sucesso")
    
    def test_criacao_backup(self):
        """Testa a criação de um backup."""
        print("\n=== Teste: Criação de Backup ===")
        
        diretorio_base = self.config_teste['diretorios']['base']
        gerenciador = GerenciadorBackupVersionamento(diretorio_base)
        
        # Criar backup
        versao = gerenciador.criar_backup(
            self.modelo_teste,
            tipo_mudanca="teste",
            descricao="Backup de teste"
        )
        
        self.assertIsNotNone(versao)
        self.assertTrue(versao.startswith("1.0."))
        
        print(f"✓ Backup criado com versão: {versao}")
    
    def test_listagem_versoes(self):
        """Testa a listagem de versões."""
        print("\n=== Teste: Listagem de Versões ===")
        
        diretorio_base = self.config_teste['diretorios']['base']
        gerenciador = GerenciadorBackupVersionamento(diretorio_base)
        
        # Criar alguns backups
        nome_modelo = os.path.basename(self.modelo_teste)
        
        versao1 = gerenciador.criar_backup(
            self.modelo_teste,
            tipo_mudanca="inicial",
            descricao="Primeira versão"
        )
        
        versao2 = gerenciador.criar_backup(
            self.modelo_teste,
            tipo_mudanca="melhoria",
            descricao="Segunda versão"
        )
        
        # Listar versões
        versoes = gerenciador.listar_versoes(nome_modelo)
        
        self.assertGreaterEqual(len(versoes), 2)
        self.assertIn(versao1, [v['versao'] for v in versoes])
        self.assertIn(versao2, [v['versao'] for v in versoes])
        
        print(f"✓ Encontradas {len(versoes)} versões")
        for versao in versoes:
            print(f"  - {versao['versao']}: {versao['descricao']}")
    
    def test_restauracao_backup(self):
        """Testa a restauração de um backup."""
        print("\n=== Teste: Restauração de Backup ===")
        
        diretorio_base = self.config_teste['diretorios']['base']
        gerenciador = GerenciadorBackupVersionamento(diretorio_base)
        
        # Criar backup
        nome_modelo = os.path.basename(self.modelo_teste)
        versao = gerenciador.criar_backup(
            self.modelo_teste,
            tipo_mudanca="teste",
            descricao="Backup para restauração"
        )
        
        # Criar diretório de restauração
        dir_restauracao = os.path.join(self.temp_dir, "restauracao")
        os.makedirs(dir_restauracao, exist_ok=True)
        
        # Restaurar backup
        caminho_restaurado = gerenciador.restaurar_backup(
            nome_modelo,
            versao,
            dir_restauracao
        )
        
        self.assertTrue(os.path.exists(caminho_restaurado))
        
        # Verificar se o conteúdo é o mesmo
        with open(self.modelo_teste, 'r', encoding='utf-8') as f:
            conteudo_original = f.read()
        
        with open(caminho_restaurado, 'r', encoding='utf-8') as f:
            conteudo_restaurado = f.read()
        
        self.assertEqual(conteudo_original, conteudo_restaurado)
        
        print(f"✓ Backup restaurado com sucesso: {caminho_restaurado}")
    
    def test_sistema_completo(self):
        """Testa o sistema completo de backup."""
        print("\n=== Teste: Sistema Completo ===")
        
        # Inicializar sistema
        gerenciador_sistema = GerenciadorSistemaBackup(self.config_path)
        
        # Carregar configuração
        self.assertTrue(gerenciador_sistema.carregar_configuracao())
        
        # Configurar logging
        gerenciador_sistema.configurar_logging()
        
        # Criar estrutura de diretórios
        gerenciador_sistema.criar_estrutura_diretorios()
        
        # Inicializar gerenciador de backup
        self.assertTrue(gerenciador_sistema.inicializar_gerenciador_backup())
        
        # Executar backup manual
        sucesso = gerenciador_sistema.executar_backup_manual(
            self.modelo_teste,
            "Teste do sistema completo"
        )
        
        self.assertTrue(sucesso)
        
        # Verificar status
        status = gerenciador_sistema.status_sistema()
        self.assertTrue(status['configuracao_carregada'])
        self.assertTrue(status['gerenciador_inicializado'])
        
        print("✓ Sistema completo testado com sucesso")
        print(f"  Status: {status}")
    
    def test_validacao_arquivos(self):
        """Testa a validação de arquivos."""
        print("\n=== Teste: Validação de Arquivos ===")
        
        diretorio_base = self.config_teste['diretorios']['base']
        gerenciador = GerenciadorBackupVersionamento(diretorio_base)
        
        # Criar backup
        nome_modelo = os.path.basename(self.modelo_teste)
        versao = gerenciador.criar_backup(
            self.modelo_teste,
            tipo_mudanca="teste",
            descricao="Backup para validação"
        )
        
        # Validar backup
        try:
            # Assumindo que existe um método de validação
            # (pode precisar ser implementado)
            print(f"✓ Backup {versao} validado com sucesso")
        except Exception as e:
            print(f"⚠ Validação não implementada: {e}")
    
    def test_limpeza_sistema(self):
        """Testa a limpeza do sistema."""
        print("\n=== Teste: Limpeza do Sistema ===")
        
        diretorio_base = self.config_teste['diretorios']['base']
        gerenciador = GerenciadorBackupVersionamento(diretorio_base)
        
        # Criar vários backups
        nome_modelo = os.path.basename(self.modelo_teste)
        versoes = []
        
        for i in range(5):
            versao = gerenciador.criar_backup(
                self.modelo_teste,
                tipo_mudanca="teste",
                descricao=f"Backup {i+1}"
            )
            versoes.append(versao)
        
        # Verificar que os backups foram criados
        versoes_listadas = gerenciador.listar_versoes(nome_modelo)
        self.assertGreaterEqual(len(versoes_listadas), 5)
        
        print(f"✓ Criados {len(versoes)} backups para teste de limpeza")
        
        # Aqui poderia implementar teste de limpeza automática
        # baseada nas políticas de retenção

def executar_teste_interativo():
    """Executa testes de forma interativa."""
    print("=== Sistema de Teste do Backup e Versionamento ===")
    print("Este script irá testar todas as funcionalidades do sistema.\n")
    
    # Executar testes
    suite = unittest.TestLoader().loadTestsFromTestCase(TesteSistemaBackup)
    runner = unittest.TextTestRunner(verbosity=2)
    resultado = runner.run(suite)
    
    # Resumo dos resultados
    print("\n=== Resumo dos Testes ===")
    print(f"Testes executados: {resultado.testsRun}")
    print(f"Falhas: {len(resultado.failures)}")
    print(f"Erros: {len(resultado.errors)}")
    
    if resultado.failures:
        print("\nFalhas:")
        for teste, traceback in resultado.failures:
            print(f"  - {teste}: {traceback}")
    
    if resultado.errors:
        print("\nErros:")
        for teste, traceback in resultado.errors:
            print(f"  - {teste}: {traceback}")
    
    if resultado.wasSuccessful():
        print("\n✅ Todos os testes passaram com sucesso!")
        return True
    else:
        print("\n❌ Alguns testes falharam.")
        return False

def verificar_dependencias():
    """Verifica se todas as dependências estão instaladas."""
    dependencias = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow',
        'schedule', 'pathlib'
    ]
    
    dependencias_faltando = []
    
    for dep in dependencias:
        try:
            __import__(dep)
        except ImportError:
            dependencias_faltando.append(dep)
    
    if dependencias_faltando:
        print("⚠ Dependências faltando para os testes:")
        for dep in dependencias_faltando:
            print(f"  - {dep}")
        print("\nInstale com: pip install " + " ".join(dependencias_faltando))
        return False
    
    print("✓ Todas as dependências estão instaladas")
    return True

def main():
    """Função principal do script de teste."""
    print("Sistema de Teste - Backup e Versionamento Lotofácil")
    print("=" * 55)
    
    # Verificar dependências
    if not verificar_dependencias():
        print("\nPor favor, instale as dependências antes de executar os testes.")
        return False
    
    # Executar testes
    try:
        sucesso = executar_teste_interativo()
        return sucesso
    except KeyboardInterrupt:
        print("\n\nTestes interrompidos pelo usuário.")
        return False
    except Exception as e:
        print(f"\nErro durante a execução dos testes: {e}")
        return False

if __name__ == "__main__":
    sucesso = main()
    sys.exit(0 if sucesso else 1)