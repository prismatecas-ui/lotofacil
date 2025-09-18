#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Validação: Integridade dos Dados Migrados

Este script valida a integridade dos dados migrados do cache JSON
para o banco SQLite, verificando consistência, completude e correção.

Autor: Sistema de Upgrade Lotofácil
Data: 2025
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Adicionar o diretório raiz ao path para importar os modelos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database_models import Base, Sorteio, NumeroSorteado, EstatisticaNumero


class MigrationValidator:
    """Classe responsável pela validação dos dados migrados."""
    
    def __init__(self, json_file_path: str, sqlite_db_path: str):
        """Inicializa o validador com os caminhos dos arquivos.
        
        Args:
            json_file_path: Caminho para o arquivo cache_concursos.json
            sqlite_db_path: Caminho para o banco SQLite
        """
        self.json_file_path = json_file_path
        self.sqlite_db_path = sqlite_db_path
        self.engine = None
        self.Session = None
        self.validation_results = []
        
    def setup_database(self) -> bool:
        """Configura a conexão com o banco SQLite."""
        try:
            if not os.path.exists(self.sqlite_db_path):
                print(f"❌ Banco SQLite não encontrado: {self.sqlite_db_path}")
                return False
            
            self.engine = create_engine(
                f'sqlite:///{self.sqlite_db_path}',
                echo=False
            )
            
            self.Session = sessionmaker(bind=self.engine)
            print(f"✅ Conexão com banco SQLite estabelecida")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao conectar com banco SQLite: {e}")
            return False
    
    def load_json_data(self) -> Dict[str, Any]:
        """Carrega os dados do arquivo JSON original."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"✅ Dados JSON carregados: {len(data)} concursos")
            return data
        except Exception as e:
            print(f"❌ Erro ao carregar JSON: {e}")
            return {}
    
    def add_result(self, test_name: str, passed: bool, message: str, details: str = ""):
        """Adiciona um resultado de validação."""
        self.validation_results.append({
            'test': test_name,
            'passed': passed,
            'message': message,
            'details': details
        })
        
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}: {message}")
        if details and not passed:
            print(f"   Detalhes: {details}")
    
    def validate_record_count(self, session, json_data: Dict[str, Any]) -> bool:
        """Valida se o número de registros migrados está correto."""
        try:
            # Contar sorteios no banco
            db_sorteios_count = session.query(Sorteio).count()
            json_sorteios_count = len(json_data)
            
            passed = db_sorteios_count == json_sorteios_count
            self.add_result(
                "Contagem de Sorteios",
                passed,
                f"DB: {db_sorteios_count}, JSON: {json_sorteios_count}",
                "" if passed else f"Diferença de {abs(db_sorteios_count - json_sorteios_count)} registros"
            )
            
            # Contar números sorteados no banco
            db_numeros_count = session.query(NumeroSorteado).count()
            expected_numeros_count = json_sorteios_count * 15  # 15 números por sorteio
            
            passed_numeros = db_numeros_count == expected_numeros_count
            self.add_result(
                "Contagem de Números",
                passed_numeros,
                f"DB: {db_numeros_count}, Esperado: {expected_numeros_count}",
                "" if passed_numeros else f"Diferença de {abs(db_numeros_count - expected_numeros_count)} números"
            )
            
            # Contar estatísticas
            db_stats_count = session.query(EstatisticaNumero).count()
            expected_stats_count = 25  # Estatísticas para números 1-25
            
            passed_stats = db_stats_count == expected_stats_count
            self.add_result(
                "Contagem de Estatísticas",
                passed_stats,
                f"DB: {db_stats_count}, Esperado: {expected_stats_count}",
                "" if passed_stats else f"Diferença de {abs(db_stats_count - expected_stats_count)} estatísticas"
            )
            
            return passed and passed_numeros and passed_stats
            
        except Exception as e:
            self.add_result("Contagem de Registros", False, f"Erro: {e}")
            return False
    
    def validate_data_integrity(self, session, json_data: Dict[str, Any]) -> bool:
        """Valida a integridade dos dados migrados."""
        try:
            errors = []
            sample_size = min(100, len(json_data))  # Validar amostra de 100 concursos
            
            # Selecionar amostra aleatória de concursos
            import random
            concursos_sample = random.sample(list(json_data.keys()), sample_size)
            
            for concurso_num in concursos_sample:
                json_concurso = json_data[concurso_num]
                
                # Buscar sorteio no banco
                db_sorteio = session.query(Sorteio).filter_by(concurso=int(concurso_num)).first()
                
                if not db_sorteio:
                    errors.append(f"Concurso {concurso_num} não encontrado no banco")
                    continue
                
                # Validar números sorteados
                json_numeros = []
                for i in range(1, 16):
                    key = f"B{i}"
                    if key in json_concurso:
                        json_numeros.append(int(json_concurso[key]))
                
                json_numeros.sort()
                db_numeros = [int(n) for n in db_sorteio.numeros_sorteados.split(',')]
                db_numeros.sort()
                
                if json_numeros != db_numeros:
                    errors.append(f"Concurso {concurso_num}: números diferentes - JSON: {json_numeros}, DB: {db_numeros}")
                
                # Validar valores de premiação
                if abs(float(json_concurso.get('Valor_Arrecadado', 0)) - (db_sorteio.valor_arrecadado or 0)) > 0.01:
                    errors.append(f"Concurso {concurso_num}: valor arrecadado diferente")
                
                if abs(int(json_concurso.get('Ganhadores_Sena', 0)) - (db_sorteio.total_ganhadores_15 or 0)) > 0:
                    errors.append(f"Concurso {concurso_num}: ganhadores sena diferentes")
            
            passed = len(errors) == 0
            self.add_result(
                "Integridade dos Dados",
                passed,
                f"Validados {sample_size} concursos",
                f"Erros encontrados: {len(errors)}" if not passed else ""
            )
            
            # Mostrar alguns erros se houver
            if errors and len(errors) <= 5:
                for error in errors:
                    print(f"   - {error}")
            elif errors:
                for error in errors[:3]:
                    print(f"   - {error}")
                print(f"   ... e mais {len(errors) - 3} erros")
            
            return passed
            
        except Exception as e:
            self.add_result("Integridade dos Dados", False, f"Erro: {e}")
            return False
    
    def validate_constraints(self, session) -> bool:
        """Valida se as constraints do banco estão sendo respeitadas."""
        try:
            errors = []
            
            # Verificar números únicos por sorteio
            from sqlalchemy import text
            duplicates_query = text("""
            SELECT sorteio_id, COUNT(*) as total
            FROM numeros_sorteados 
            GROUP BY sorteio_id, numero 
            HAVING COUNT(*) > 1
            """)
            
            duplicates = session.execute(duplicates_query).fetchall()
            if duplicates:
                errors.append(f"Encontrados {len(duplicates)} números duplicados em sorteios")
            
            # Verificar se todos os sorteios têm exatamente 15 números
            count_query = text("""
            SELECT sorteio_id, COUNT(*) as total
            FROM numeros_sorteados 
            GROUP BY sorteio_id 
            HAVING COUNT(*) != 15
            """)
            
            wrong_counts = session.execute(count_query).fetchall()
            if wrong_counts:
                errors.append(f"Encontrados {len(wrong_counts)} sorteios com número incorreto de bolas")
            
            # Verificar números fora do range 1-25
            invalid_numbers = session.query(NumeroSorteado).filter(
                (NumeroSorteado.numero < 1) | (NumeroSorteado.numero > 25)
            ).count()
            
            if invalid_numbers > 0:
                errors.append(f"Encontrados {invalid_numbers} números fora do range 1-25")
            
            # Verificar posições fora do range 1-15
            invalid_positions = session.query(NumeroSorteado).filter(
                (NumeroSorteado.posicao < 1) | (NumeroSorteado.posicao > 15)
            ).count()
            
            if invalid_positions > 0:
                errors.append(f"Encontradas {invalid_positions} posições inválidas")
            
            passed = len(errors) == 0
            self.add_result(
                "Validação de Constraints",
                passed,
                "Todas as constraints respeitadas" if passed else f"{len(errors)} violações encontradas",
                "; ".join(errors) if errors else ""
            )
            
            return passed
            
        except Exception as e:
            self.add_result("Validação de Constraints", False, f"Erro: {e}")
            return False
    
    def validate_statistics(self, session) -> bool:
        """Valida se as estatísticas foram calculadas corretamente."""
        try:
            errors = []
            
            # Verificar se todas as estatísticas foram criadas (1-25)
            stats_count = session.query(EstatisticaNumero).count()
            if stats_count != 25:
                errors.append(f"Esperadas 25 estatísticas, encontradas {stats_count}")
            
            # Validar algumas estatísticas calculadas
            for numero in [1, 13, 25]:  # Testar alguns números
                stat = session.query(EstatisticaNumero).filter_by(numero=numero).first()
                if not stat:
                    errors.append(f"Estatística do número {numero} não encontrada")
                    continue
                
                # Contar frequência real no banco
                real_freq = session.query(NumeroSorteado).filter_by(numero=numero).count()
                
                if stat.frequencia_absoluta != real_freq:
                    errors.append(f"Número {numero}: frequência incorreta (stat: {stat.frequencia_absoluta}, real: {real_freq})")
            
            # Verificar se a soma das frequências bate com o total esperado
            total_sorteios = session.query(Sorteio).count()
            total_numeros_esperado = total_sorteios * 15
            
            soma_frequencias = session.query(func.sum(EstatisticaNumero.frequencia_absoluta)).scalar() or 0
            
            if abs(soma_frequencias - total_numeros_esperado) > 0:
                errors.append(f"Soma das frequências ({soma_frequencias}) != total esperado ({total_numeros_esperado})")
            
            passed = len(errors) == 0
            self.add_result(
                "Validação de Estatísticas",
                passed,
                "Estatísticas corretas" if passed else f"{len(errors)} problemas encontrados",
                "; ".join(errors) if errors else ""
            )
            
            return passed
            
        except Exception as e:
            self.add_result("Validação de Estatísticas", False, f"Erro: {e}")
            return False
    
    def validate_performance(self, session) -> bool:
        """Testa a performance de consultas básicas."""
        try:
            import time
            
            # Teste 1: Consulta de sorteio por concurso
            start_time = time.time()
            sorteio = session.query(Sorteio).filter_by(concurso=1000).first()
            query1_time = time.time() - start_time
            
            # Teste 2: Consulta de números de um sorteio
            start_time = time.time()
            numeros = session.query(NumeroSorteado).filter_by(concurso=1000).all()
            query2_time = time.time() - start_time
            
            # Teste 3: Consulta de estatística
            start_time = time.time()
            stat = session.query(EstatisticaNumero).filter_by(numero=13).first()
            query3_time = time.time() - start_time
            
            # Teste 4: Contagem total
            start_time = time.time()
            total = session.query(Sorteio).count()
            query4_time = time.time() - start_time
            
            max_acceptable_time = 1.0  # 1 segundo
            
            all_fast = all([
                query1_time < max_acceptable_time,
                query2_time < max_acceptable_time,
                query3_time < max_acceptable_time,
                query4_time < max_acceptable_time
            ])
            
            self.add_result(
                "Performance de Consultas",
                all_fast,
                f"Tempos: {query1_time:.3f}s, {query2_time:.3f}s, {query3_time:.3f}s, {query4_time:.3f}s",
                "Algumas consultas estão lentas" if not all_fast else ""
            )
            
            return all_fast
            
        except Exception as e:
            self.add_result("Performance de Consultas", False, f"Erro: {e}")
            return False
    
    def run_validation(self) -> bool:
        """Executa todas as validações."""
        print("🔍 Iniciando validação da migração")
        print("=" * 50)
        
        # Configurar banco
        if not self.setup_database():
            return False
        
        # Carregar dados JSON
        json_data = self.load_json_data()
        if not json_data:
            return False
        
        # Executar validações
        session = self.Session()
        all_passed = True
        
        try:
            print("\n📊 Executando validações...")
            
            # Validação 1: Contagem de registros
            all_passed &= self.validate_record_count(session, json_data)
            
            # Validação 2: Integridade dos dados
            all_passed &= self.validate_data_integrity(session, json_data)
            
            # Validação 3: Constraints do banco
            all_passed &= self.validate_constraints(session)
            
            # Validação 4: Estatísticas
            all_passed &= self.validate_statistics(session)
            
            # Validação 5: Performance
            all_passed &= self.validate_performance(session)
            
            # Resumo final
            print("\n" + "=" * 50)
            print("📋 RESUMO DA VALIDAÇÃO")
            print("=" * 50)
            
            passed_count = sum(1 for r in self.validation_results if r['passed'])
            total_count = len(self.validation_results)
            
            for result in self.validation_results:
                status = "✅" if result['passed'] else "❌"
                print(f"{status} {result['test']}: {result['message']}")
            
            print("\n" + "=" * 50)
            if all_passed:
                print("🎉 VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
                print(f"✅ Todos os {total_count} testes passaram")
                print("\n📋 A migração foi realizada corretamente:")
                print("   • Todos os dados foram migrados")
                print("   • Integridade dos dados mantida")
                print("   • Constraints respeitadas")
                print("   • Estatísticas calculadas corretamente")
                print("   • Performance adequada")
            else:
                print("⚠️ VALIDAÇÃO CONCLUÍDA COM PROBLEMAS")
                print(f"✅ {passed_count}/{total_count} testes passaram")
                print(f"❌ {total_count - passed_count} testes falharam")
                print("\n📋 Revise os problemas encontrados acima")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Erro durante validação: {e}")
            return False
            
        finally:
            session.close()


def main():
    """Função principal do script."""
    # Caminhos dos arquivos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(base_dir, 'base', 'cache_concursos.json')
    sqlite_db = os.path.join(base_dir, 'database', 'lotofacil.db')
    
    # Criar validador e executar
    validator = MigrationValidator(json_file, sqlite_db)
    success = validator.run_validation()
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)