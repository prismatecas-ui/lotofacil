#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Backup: Base de Dados Lotofácil

Este script cria um backup completo da base de dados atual,
incluindo banco SQLite, cache JSON, Excel e arquivos de configuração.

Autor: Sistema de Upgrade Lotofácil
Data: 2025
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


class LotofacilBackup:
    """Classe responsável pelo backup da base de dados Lotofácil."""
    
    def __init__(self, base_dir: str = None):
        """Inicializa o sistema de backup.
        
        Args:
            base_dir: Diretório base do projeto (padrão: diretório atual)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.base_dir / "backups" / f"backup_{self.timestamp}"
        
        # Arquivos e diretórios importantes para backup
        self.backup_items = [
            # Banco de dados
            ("database/lotofacil.db", "database/"),
            
            # Cache e dados
            ("base/cache_concursos.json", "base/"),
            ("base/base_dados.xlsx", "base/"),
            ("base/resultados.csv", "base/"),
            
            # Scripts de migração
            ("migrate_json_to_sqlite.py", ""),
            ("validate_migration.py", ""),
            ("update_excel_data.py", ""),
            
            # Modelos e configurações
            ("models/database_models.py", "models/"),
            ("database/create_tables.sql", "database/"),
            
            # Configurações
            ("config/", "config/"),
            
            # Logs importantes
            ("logs/", "logs/"),
        ]
    
    def create_backup_directory(self) -> bool:
        """Cria o diretório de backup."""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Diretório de backup criado: {self.backup_dir}")
            return True
        except Exception as e:
            print(f"❌ Erro ao criar diretório de backup: {e}")
            return False
    
    def backup_file(self, source_path: str, dest_subdir: str = "") -> bool:
        """Faz backup de um arquivo específico.
        
        Args:
            source_path: Caminho do arquivo/diretório fonte
            dest_subdir: Subdiretório de destino no backup
            
        Returns:
            True se o backup foi bem-sucedido
        """
        try:
            source = self.base_dir / source_path
            
            if not source.exists():
                print(f"⚠️ Arquivo não encontrado: {source_path}")
                return False
            
            # Definir destino
            if dest_subdir:
                dest_dir = self.backup_dir / dest_subdir
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / source.name
            else:
                dest = self.backup_dir / source.name
            
            # Copiar arquivo ou diretório
            if source.is_file():
                shutil.copy2(source, dest)
                file_size = source.stat().st_size
                print(f"✅ Arquivo copiado: {source_path} ({file_size:,} bytes)")
            elif source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
                print(f"✅ Diretório copiado: {source_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao fazer backup de {source_path}: {e}")
            return False
    
    def create_backup_info(self) -> bool:
        """Cria arquivo com informações do backup."""
        try:
            info_file = self.backup_dir / "backup_info.txt"
            
            # Coletar informações do sistema
            db_path = self.base_dir / "database" / "lotofacil.db"
            cache_path = self.base_dir / "base" / "cache_concursos.json"
            excel_path = self.base_dir / "base" / "base_dados.xlsx"
            
            info_content = f"""BACKUP LOTOFÁCIL - INFORMAÇÕES
{'=' * 50}

Data/Hora do Backup: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Diretório Base: {self.base_dir}
Diretório Backup: {self.backup_dir}

ARQUIVOS INCLUÍDOS:
{'-' * 30}
"""
            
            # Informações dos arquivos principais
            if db_path.exists():
                db_size = db_path.stat().st_size
                info_content += f"• Banco SQLite: {db_size:,} bytes\n"
            
            if cache_path.exists():
                cache_size = cache_path.stat().st_size
                info_content += f"• Cache JSON: {cache_size:,} bytes\n"
            
            if excel_path.exists():
                excel_size = excel_path.stat().st_size
                info_content += f"• Base Excel: {excel_size:,} bytes\n"
            
            # Listar todos os arquivos do backup
            info_content += f"\nARQUIVOS NO BACKUP:\n{'-' * 30}\n"
            
            for item_path, _ in self.backup_items:
                source = self.base_dir / item_path
                if source.exists():
                    if source.is_file():
                        size = source.stat().st_size
                        info_content += f"• {item_path} ({size:,} bytes)\n"
                    else:
                        info_content += f"• {item_path}/ (diretório)\n"
            
            info_content += f"\nESTATÍSTICAS:\n{'-' * 30}\n"
            
            # Estatísticas do banco (se existir)
            if db_path.exists():
                try:
                    import sqlite3
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    
                    # Contar registros
                    cursor.execute("SELECT COUNT(*) FROM sorteios")
                    sorteios_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM numeros_sorteados")
                    numeros_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM estatisticas_numeros")
                    stats_count = cursor.fetchone()[0]
                    
                    info_content += f"• Sorteios no banco: {sorteios_count:,}\n"
                    info_content += f"• Números sorteados: {numeros_count:,}\n"
                    info_content += f"• Estatísticas: {stats_count:,}\n"
                    
                    conn.close()
                    
                except Exception as e:
                    info_content += f"• Erro ao ler estatísticas do banco: {e}\n"
            
            # Estatísticas do cache (se existir)
            if cache_path.exists():
                try:
                    import json
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    info_content += f"• Concursos no cache: {len(cache_data):,}\n"
                except Exception as e:
                    info_content += f"• Erro ao ler cache: {e}\n"
            
            info_content += f"\nCOMO RESTAURAR:\n{'-' * 30}\n"
            info_content += "1. Pare todos os processos que usam a base\n"
            info_content += "2. Copie os arquivos de volta para suas localizações originais\n"
            info_content += "3. Verifique as permissões dos arquivos\n"
            info_content += "4. Reinicie os serviços necessários\n"
            
            # Escrever arquivo de informações
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(info_content)
            
            print(f"✅ Arquivo de informações criado: backup_info.txt")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao criar arquivo de informações: {e}")
            return False
    
    def create_zip_backup(self) -> bool:
        """Cria um arquivo ZIP com todo o backup."""
        try:
            zip_path = self.backup_dir.parent / f"backup_{self.timestamp}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Adicionar todos os arquivos do diretório de backup
                for root, dirs, files in os.walk(self.backup_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arc_name = file_path.relative_to(self.backup_dir.parent)
                        zipf.write(file_path, arc_name)
            
            zip_size = zip_path.stat().st_size
            print(f"✅ Backup ZIP criado: {zip_path.name} ({zip_size:,} bytes)")
            
            # Remover diretório temporário
            shutil.rmtree(self.backup_dir)
            print(f"✅ Diretório temporário removido")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao criar ZIP: {e}")
            return False
    
    def run_backup(self, create_zip: bool = True) -> bool:
        """Executa o backup completo.
        
        Args:
            create_zip: Se deve criar arquivo ZIP (padrão: True)
            
        Returns:
            True se o backup foi bem-sucedido
        """
        print("💾 Iniciando backup da base Lotofácil")
        print("=" * 50)
        
        # Criar diretório de backup
        if not self.create_backup_directory():
            return False
        
        # Fazer backup dos arquivos
        success_count = 0
        total_count = len(self.backup_items)
        
        print(f"\n📁 Fazendo backup de {total_count} itens...")
        
        for source_path, dest_subdir in self.backup_items:
            if self.backup_file(source_path, dest_subdir):
                success_count += 1
        
        # Criar arquivo de informações
        if self.create_backup_info():
            success_count += 1
            total_count += 1
        
        print(f"\n📊 Backup concluído: {success_count}/{total_count} itens")
        
        # Criar ZIP se solicitado
        if create_zip:
            print(f"\n🗜️ Compactando backup...")
            if not self.create_zip_backup():
                return False
        
        print("\n" + "=" * 50)
        print("🎉 BACKUP CONCLUÍDO COM SUCESSO!")
        print("=" * 50)
        print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"📁 Localização: backups/backup_{self.timestamp}.zip")
        print(f"✅ {success_count}/{total_count} itens salvos")
        
        if success_count == total_count:
            print("\n📋 Backup completo realizado com sucesso!")
            print("   • Banco SQLite incluído")
            print("   • Cache JSON incluído")
            print("   • Base Excel incluída")
            print("   • Scripts de migração incluídos")
            print("   • Configurações incluídas")
            return True
        else:
            print(f"\n⚠️ Backup parcial: {total_count - success_count} itens falharam")
            return False


def main():
    """Função principal do script."""
    # Criar sistema de backup
    backup_system = LotofacilBackup()
    
    # Executar backup
    success = backup_system.run_backup(create_zip=True)
    
    return success


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)