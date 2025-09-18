#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Migração: Cache JSON para SQLite

Este script migra os dados dos concursos da Lotofácil do arquivo
cache_concursos.json para o banco de dados SQLite usando SQLAlchemy.

Autor: Sistema de Upgrade Lotofácil
Data: 2025
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Adicionar o diretório raiz ao path para importar os modelos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database_models import Base, Sorteio, NumeroSorteado, EstatisticaNumero


class JsonToSqliteMigrator:
    """Classe responsável pela migração dos dados do JSON para SQLite."""
    
    def __init__(self, json_file_path: str, sqlite_db_path: str):
        """Inicializa o migrador com os caminhos dos arquivos.
        
        Args:
            json_file_path: Caminho para o arquivo cache_concursos.json
            sqlite_db_path: Caminho para o banco SQLite de destino
        """
        self.json_file_path = json_file_path
        self.sqlite_db_path = sqlite_db_path
        self.engine = None
        self.Session = None
        
    def setup_database(self):
        """Configura a conexão com o banco SQLite e cria as tabelas."""
        try:
            # Criar diretório do banco se não existir
            db_dir = os.path.dirname(self.sqlite_db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Configurar engine SQLite
            self.engine = create_engine(
                f'sqlite:///{self.sqlite_db_path}',
                echo=False,  # Mudar para True para debug SQL
                pool_pre_ping=True
            )
            
            # Criar todas as tabelas
            Base.metadata.create_all(self.engine)
            
            # Configurar session factory
            self.Session = sessionmaker(bind=self.engine)
            
            print(f"✅ Banco SQLite configurado: {self.sqlite_db_path}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao configurar banco SQLite: {e}")
            return False
    
    def load_json_data(self) -> Dict[str, Any]:
        """Carrega os dados do arquivo JSON.
        
        Returns:
            Dicionário com os dados dos concursos
        """
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            print(f"✅ Dados JSON carregados: {len(data)} concursos")
            return data
            
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {self.json_file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Erro ao decodificar JSON: {e}")
            return {}
        except Exception as e:
            print(f"❌ Erro ao carregar JSON: {e}")
            return {}
    
    def parse_date(self, date_str: str) -> datetime:
        """Converte string de data do formato DD/MM/YYYY para datetime.
        
        Args:
            date_str: Data no formato "DD/MM/YYYY"
            
        Returns:
            Objeto datetime
        """
        try:
            return datetime.strptime(date_str, '%d/%m/%Y')
        except ValueError:
            # Fallback para outros formatos possíveis
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                print(f"⚠️ Formato de data inválido: {date_str}")
                return datetime.now()
    
    def extract_numbers_from_concurso(self, concurso_data: Dict[str, Any]) -> List[int]:
        """Extrai os números sorteados de um concurso.
        
        Args:
            concurso_data: Dados do concurso do JSON
            
        Returns:
            Lista com os 15 números sorteados
        """
        numeros = []
        for i in range(1, 16):  # B1 a B15
            key = f"B{i}"
            if key in concurso_data:
                numeros.append(int(concurso_data[key]))
        
        # Ordenar os números
        numeros.sort()
        return numeros
    
    def migrate_sorteio(self, session, concurso_num: str, concurso_data: Dict[str, Any]) -> bool:
        """Migra um único sorteio para o banco SQLite.
        
        Args:
            session: Sessão SQLAlchemy
            concurso_num: Número do concurso
            concurso_data: Dados do concurso
            
        Returns:
            True se a migração foi bem-sucedida
        """
        try:
            # Verificar se o sorteio já existe
            existing = session.query(Sorteio).filter_by(concurso=int(concurso_num)).first()
            if existing:
                print(f"⚠️ Concurso {concurso_num} já existe, pulando...")
                return True
            
            # Extrair números sorteados
            numeros = self.extract_numbers_from_concurso(concurso_data)
            if len(numeros) != 15:
                print(f"❌ Concurso {concurso_num}: números inválidos ({len(numeros)} números)")
                return False
            
            # Criar objeto Sorteio
            sorteio = Sorteio(
                concurso=int(concurso_num),
                data_sorteio=self.parse_date(concurso_data.get('Data Sorteio', '')),
                numeros_sorteados=','.join(map(str, numeros)),
                valor_arrecadado=float(concurso_data.get('Valor_Arrecadado', 0.0)),
                valor_acumulado=float(concurso_data.get('Valor_Acumulado', 0.0)),
                valor_estimado_proximo=float(concurso_data.get('Estimativa_Premio', 0.0)),
                total_ganhadores_15=int(concurso_data.get('Ganhadores_Sena', 0)),
                total_ganhadores_14=int(concurso_data.get('Ganhadores_Quina', 0)),
                total_ganhadores_13=int(concurso_data.get('Ganhadores_Quadra', 0)),
                total_ganhadores_12=int(concurso_data.get('Ganhadores_Terno', 0)),
                total_ganhadores_11=int(concurso_data.get('Ganhadores_Duque', 0)),
                valor_premio_15=float(concurso_data.get('Rateio_Sena', 0.0)),
                valor_premio_14=float(concurso_data.get('Rateio_Quina', 0.0)),
                valor_premio_13=float(concurso_data.get('Rateio_Quadra', 0.0)),
                valor_premio_12=float(concurso_data.get('Rateio_Terno', 0.0)),
                valor_premio_11=float(concurso_data.get('Rateio_Duque', 0.0))
            )
            
            # Adicionar à sessão
            session.add(sorteio)
            session.flush()  # Para obter o ID
            
            # Criar registros de números sorteados
            for posicao, numero in enumerate(numeros, 1):
                numero_sorteado = NumeroSorteado(
                    sorteio_id=sorteio.id,
                    concurso=int(concurso_num),
                    numero=numero,
                    posicao=posicao
                )
                session.add(numero_sorteado)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao migrar concurso {concurso_num}: {e}")
            return False
    
    def update_estatisticas(self, session):
        """Atualiza as estatísticas dos números após a migração."""
        try:
            print("📊 Atualizando estatísticas dos números...")
            
            # Limpar estatísticas existentes
            session.query(EstatisticaNumero).delete()
            
            # Calcular estatísticas para cada número (1 a 25)
            for numero in range(1, 26):
                # Contar frequência absoluta
                freq_absoluta = session.query(NumeroSorteado).filter_by(numero=numero).count()
                
                # Obter total de sorteios
                total_sorteios = session.query(Sorteio).count()
                
                # Calcular frequência relativa
                freq_relativa = (freq_absoluta / total_sorteios * 100) if total_sorteios > 0 else 0.0
                
                # Obter último sorteio do número
                ultimo_registro = session.query(NumeroSorteado).filter_by(numero=numero).order_by(NumeroSorteado.concurso.desc()).first()
                ultimo_sorteio = ultimo_registro.concurso if ultimo_registro else 0
                
                # Calcular atraso atual
                ultimo_concurso = session.query(Sorteio.concurso).order_by(Sorteio.concurso.desc()).first()
                atraso_atual = (ultimo_concurso[0] - ultimo_sorteio) if ultimo_concurso and ultimo_sorteio > 0 else 0
                
                # Criar estatística
                estatistica = EstatisticaNumero(
                    numero=numero,
                    total_sorteios=total_sorteios,
                    frequencia_absoluta=freq_absoluta,
                    frequencia_relativa=freq_relativa,
                    ultimo_sorteio=ultimo_sorteio,
                    atraso_atual=atraso_atual
                )
                
                session.add(estatistica)
            
            session.commit()
            print("✅ Estatísticas atualizadas com sucesso")
            
        except Exception as e:
            print(f"❌ Erro ao atualizar estatísticas: {e}")
            session.rollback()
    
    def run_migration(self) -> bool:
        """Executa a migração completa dos dados.
        
        Returns:
            True se a migração foi bem-sucedida
        """
        print("🚀 Iniciando migração JSON → SQLite")
        print("=" * 50)
        
        # Configurar banco
        if not self.setup_database():
            return False
        
        # Carregar dados JSON
        json_data = self.load_json_data()
        if not json_data:
            return False
        
        # Migrar dados
        session = self.Session()
        migrated_count = 0
        error_count = 0
        
        try:
            # Ordenar concursos por número
            concursos_ordenados = sorted(json_data.keys(), key=int)
            total_concursos = len(concursos_ordenados)
            
            print(f"📋 Migrando {total_concursos} concursos...")
            
            for i, concurso_num in enumerate(concursos_ordenados, 1):
                concurso_data = json_data[concurso_num]
                
                if self.migrate_sorteio(session, concurso_num, concurso_data):
                    migrated_count += 1
                else:
                    error_count += 1
                
                # Commit a cada 100 registros
                if i % 100 == 0:
                    session.commit()
                    print(f"📈 Progresso: {i}/{total_concursos} ({i/total_concursos*100:.1f}%)")
            
            # Commit final
            session.commit()
            
            # Atualizar estatísticas
            self.update_estatisticas(session)
            
            print("\n" + "=" * 50)
            print(f"✅ Migração concluída!")
            print(f"📊 Concursos migrados: {migrated_count}")
            print(f"❌ Erros encontrados: {error_count}")
            print(f"💾 Banco SQLite: {self.sqlite_db_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro durante a migração: {e}")
            session.rollback()
            return False
            
        finally:
            session.close()


def main():
    """Função principal do script."""
    # Caminhos dos arquivos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(base_dir, 'base', 'cache_concursos.json')
    sqlite_db = os.path.join(base_dir, 'database', 'lotofacil.db')
    
    # Verificar se o arquivo JSON existe
    if not os.path.exists(json_file):
        print(f"❌ Arquivo JSON não encontrado: {json_file}")
        return False
    
    # Criar migrador e executar
    migrator = JsonToSqliteMigrator(json_file, sqlite_db)
    success = migrator.run_migration()
    
    if success:
        print("\n🎉 Migração realizada com sucesso!")
        print("\n📋 Próximos passos:")
        print("   1. Validar integridade dos dados migrados")
        print("   2. Executar testes de consulta")
        print("   3. Criar backup do banco SQLite")
    else:
        print("\n💥 Migração falhou. Verifique os erros acima.")


if __name__ == "__main__":
    main()