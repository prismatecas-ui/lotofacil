#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface para Informações de Sorteios - Lotofácil

Este módulo implementa uma interface completa para consulta e exibição
de informações detalhadas sobre sorteios da Lotofácil.

Autor: Sistema Lotofácil
Versão: 1.0.0
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import requests

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterfaceSorteios:
    """
    Interface completa para informações de sorteios da Lotofácil.
    
    Funcionalidades:
    - Consulta de resultados por período
    - Análise estatística de sorteios
    - Geração de relatórios visuais
    - Comparação entre concursos
    - Busca por padrões específicos
    """
    
    def __init__(self, db_path: str = "database/lotofacil.db"):
        """
        Inicializa a interface de sorteios.
        
        Args:
            db_path: Caminho para o banco de dados SQLite
        """
        self.db_path = db_path
        self.numeros_lotofacil = list(range(1, 26))
        
    def conectar_db(self) -> sqlite3.Connection:
        """Estabelece conexão com o banco de dados."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            logger.error(f"Erro ao conectar com o banco: {e}")
            raise
    
    def consultar_concurso(self, numero_concurso: int) -> Dict:
        """
        Consulta informações detalhadas de um concurso específico.
        
        Args:
            numero_concurso: Número do concurso
            
        Returns:
            Dict com informações completas do concurso
        """
        try:
            conn = self.conectar_db()
            
            query = """
            SELECT * FROM concursos 
            WHERE numero = ?
            """
            
            resultado = pd.read_sql_query(query, conn, params=(numero_concurso,))
            conn.close()
            
            if resultado.empty:
                return {'erro': f'Concurso {numero_concurso} não encontrado'}
            
            concurso = resultado.iloc[0]
            numeros_sorteados = json.loads(concurso['numeros_sorteados'])
            
            # Análise detalhada dos números
            analise_numeros = self._analisar_numeros_sorteados(numeros_sorteados)
            
            # Comparar com concursos anteriores
            comparacao = self._comparar_com_anteriores(numero_concurso, numeros_sorteados)
            
            return {
                'concurso': int(concurso['concurso']),
                'data_sorteio': concurso['data_sorteio'],
                'numeros_sorteados': numeros_sorteados,
                'analise_numeros': analise_numeros,
                'comparacao_anteriores': comparacao,
                'premio_total': concurso.get('premio_total', 'N/A'),
                'ganhadores_15': concurso.get('ganhadores_15', 'N/A'),
                'ganhadores_14': concurso.get('ganhadores_14', 'N/A'),
                'ganhadores_13': concurso.get('ganhadores_13', 'N/A'),
                'ganhadores_12': concurso.get('ganhadores_12', 'N/A'),
                'ganhadores_11': concurso.get('ganhadores_11', 'N/A')
            }
            
        except Exception as e:
            logger.error(f"Erro ao consultar concurso {numero_concurso}: {e}")
            return {'erro': str(e)}
    
    def _analisar_numeros_sorteados(self, numeros: List[int]) -> Dict:
        """
        Realiza análise detalhada dos números sorteados.
        """
        # Análise de paridade
        pares = [n for n in numeros if n % 2 == 0]
        impares = [n for n in numeros if n % 2 != 0]
        
        # Análise de sequências
        numeros_ordenados = sorted(numeros)
        sequencias = self._encontrar_sequencias(numeros_ordenados)
        
        # Análise por dezenas
        primeira_dezena = [n for n in numeros if 1 <= n <= 10]
        segunda_dezena = [n for n in numeros if 11 <= n <= 20]
        terceira_dezena = [n for n in numeros if 21 <= n <= 25]
        
        # Análise por linhas da cartela (5x5)
        linhas = {
            'linha_1': [n for n in numeros if 1 <= n <= 5],
            'linha_2': [n for n in numeros if 6 <= n <= 10],
            'linha_3': [n for n in numeros if 11 <= n <= 15],
            'linha_4': [n for n in numeros if 16 <= n <= 20],
            'linha_5': [n for n in numeros if 21 <= n <= 25]
        }
        
        # Análise por colunas da cartela
        colunas = {
            'coluna_1': [n for n in numeros if n in [1, 6, 11, 16, 21]],
            'coluna_2': [n for n in numeros if n in [2, 7, 12, 17, 22]],
            'coluna_3': [n for n in numeros if n in [3, 8, 13, 18, 23]],
            'coluna_4': [n for n in numeros if n in [4, 9, 14, 19, 24]],
            'coluna_5': [n for n in numeros if n in [5, 10, 15, 20, 25]]
        }
        
        return {
            'total_numeros': len(numeros),
            'pares': {'numeros': pares, 'quantidade': len(pares)},
            'impares': {'numeros': impares, 'quantidade': len(impares)},
            'sequencias': sequencias,
            'distribuicao_dezenas': {
                'primeira_dezena': {'numeros': primeira_dezena, 'quantidade': len(primeira_dezena)},
                'segunda_dezena': {'numeros': segunda_dezena, 'quantidade': len(segunda_dezena)},
                'terceira_dezena': {'numeros': terceira_dezena, 'quantidade': len(terceira_dezena)}
            },
            'distribuicao_linhas': {k: {'numeros': v, 'quantidade': len(v)} for k, v in linhas.items()},
            'distribuicao_colunas': {k: {'numeros': v, 'quantidade': len(v)} for k, v in colunas.items()},
            'menor_numero': min(numeros),
            'maior_numero': max(numeros),
            'amplitude': max(numeros) - min(numeros),
            'soma_total': sum(numeros),
            'media': round(sum(numeros) / len(numeros), 2)
        }
    
    def _encontrar_sequencias(self, numeros_ordenados: List[int]) -> List[List[int]]:
        """
        Encontra sequências consecutivas nos números sorteados.
        """
        sequencias = []
        sequencia_atual = [numeros_ordenados[0]]
        
        for i in range(1, len(numeros_ordenados)):
            if numeros_ordenados[i] == numeros_ordenados[i-1] + 1:
                sequencia_atual.append(numeros_ordenados[i])
            else:
                if len(sequencia_atual) >= 2:
                    sequencias.append(sequencia_atual.copy())
                sequencia_atual = [numeros_ordenados[i]]
        
        # Verificar a última sequência
        if len(sequencia_atual) >= 2:
            sequencias.append(sequencia_atual)
        
        return sequencias
    
    def _comparar_com_anteriores(self, numero_concurso: int, numeros_atuais: List[int]) -> Dict:
        """
        Compara o concurso atual com os anteriores.
        """
        try:
            conn = self.conectar_db()
            
            # Buscar últimos 10 concursos anteriores
            query = """
            SELECT concurso, numeros_sorteados 
            FROM concursos 
            WHERE numero < ? 
            ORDER BY numero DESC 
            LIMIT 10
            """
            
            anteriores = pd.read_sql_query(query, conn, params=(numero_concurso,))
            conn.close()
            
            if anteriores.empty:
                return {'erro': 'Nenhum concurso anterior encontrado'}
            
            comparacoes = []
            
            for _, anterior in anteriores.iterrows():
                numeros_anterior = json.loads(anterior['numeros_sorteados'])
                
                # Calcular coincidências
                coincidencias = set(numeros_atuais).intersection(set(numeros_anterior))
                
                comparacoes.append({
                    'concurso': int(anterior['concurso']),
                    'numeros_coincidentes': list(coincidencias),
                    'total_coincidencias': len(coincidencias),
                    'numeros_diferentes': list(set(numeros_atuais) - coincidencias)
                })
            
            # Estatísticas gerais
            total_coincidencias = [comp['total_coincidencias'] for comp in comparacoes]
            
            return {
                'comparacoes_individuais': comparacoes,
                'estatisticas': {
                    'media_coincidencias': round(np.mean(total_coincidencias), 2),
                    'max_coincidencias': max(total_coincidencias),
                    'min_coincidencias': min(total_coincidencias),
                    'concursos_analisados': len(comparacoes)
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na comparação: {e}")
            return {'erro': str(e)}
    
    def consultar_periodo(self, 
                         data_inicio: str, 
                         data_fim: str, 
                         incluir_analise: bool = True) -> Dict:
        """
        Consulta concursos em um período específico.
        
        Args:
            data_inicio: Data de início (formato: YYYY-MM-DD)
            data_fim: Data de fim (formato: YYYY-MM-DD)
            incluir_analise: Se deve incluir análise estatística
            
        Returns:
            Dict com concursos do período e análises
        """
        try:
            conn = self.conectar_db()
            
            query = """
            SELECT * FROM concursos 
            WHERE data_sorteio BETWEEN ? AND ?
            ORDER BY numero
            """
            
            resultados = pd.read_sql_query(query, conn, params=(data_inicio, data_fim))
            conn.close()
            
            if resultados.empty:
                return {'erro': f'Nenhum concurso encontrado no período {data_inicio} a {data_fim}'}
            
            concursos = []
            todos_numeros = []
            
            for _, resultado in resultados.iterrows():
                numeros_sorteados = json.loads(resultado['numeros_sorteados'])
                todos_numeros.extend(numeros_sorteados)
                
                concurso_info = {
                    'concurso': int(resultado['concurso']),
                    'data_sorteio': resultado['data_sorteio'],
                    'numeros_sorteados': numeros_sorteados
                }
                
                if incluir_analise:
                    concurso_info['analise'] = self._analisar_numeros_sorteados(numeros_sorteados)
                
                concursos.append(concurso_info)
            
            resposta = {
                'periodo': {'inicio': data_inicio, 'fim': data_fim},
                'total_concursos': len(concursos),
                'concursos': concursos
            }
            
            if incluir_analise:
                resposta['analise_periodo'] = self._analisar_periodo(todos_numeros, len(concursos))
            
            return resposta
            
        except Exception as e:
            logger.error(f"Erro ao consultar período: {e}")
            return {'erro': str(e)}
    
    def _analisar_periodo(self, todos_numeros: List[int], total_concursos: int) -> Dict:
        """
        Analisa estatísticas de um período de concursos.
        """
        # Frequência de cada número
        frequencia = Counter(todos_numeros)
        
        # Números mais e menos sorteados
        mais_sorteados = frequencia.most_common(5)
        menos_sorteados = frequencia.most_common()[:-6:-1]  # Últimos 5
        
        # Números que não saíram
        numeros_nao_sorteados = [n for n in self.numeros_lotofacil if n not in frequencia]
        
        # Estatísticas de frequência
        frequencias = list(frequencia.values())
        
        return {
            'frequencia_numeros': dict(frequencia),
            'mais_sorteados': mais_sorteados,
            'menos_sorteados': menos_sorteados,
            'numeros_nao_sorteados': numeros_nao_sorteados,
            'estatisticas_frequencia': {
                'media_aparicoes': round(np.mean(frequencias), 2),
                'max_aparicoes': max(frequencias),
                'min_aparicoes': min(frequencias),
                'desvio_padrao': round(np.std(frequencias), 2)
            },
            'total_sorteios': len(todos_numeros),
            'media_por_concurso': round(len(todos_numeros) / total_concursos, 2)
        }
    
    def buscar_padroes(self, 
                      padrao: Dict, 
                      limite_resultados: int = 50) -> Dict:
        """
        Busca concursos que atendem a padrões específicos.
        
        Args:
            padrao: Dicionário com critérios de busca
            limite_resultados: Número máximo de resultados
            
        Returns:
            Dict com concursos que atendem ao padrão
        """
        try:
            conn = self.conectar_db()
            
            query = "SELECT * FROM concursos ORDER BY numero DESC"
            if limite_resultados:
                query += f" LIMIT {limite_resultados * 3}"  # Buscar mais para filtrar
            
            resultados = pd.read_sql_query(query, conn)
            conn.close()
            
            concursos_encontrados = []
            
            for _, resultado in resultados.iterrows():
                numeros_sorteados = json.loads(resultado['numeros_sorteados'])
                
                if self._verificar_padrao(numeros_sorteados, padrao):
                    concursos_encontrados.append({
                        'concurso': int(resultado['numero']),
                        'data_sorteio': resultado['data_sorteio'],
                        'numeros_sorteados': numeros_sorteados,
                        'analise': self._analisar_numeros_sorteados(numeros_sorteados)
                    })
                    
                    if len(concursos_encontrados) >= limite_resultados:
                        break
            
            return {
                'padrao_buscado': padrao,
                'total_encontrados': len(concursos_encontrados),
                'concursos': concursos_encontrados
            }
            
        except Exception as e:
            logger.error(f"Erro na busca por padrões: {e}")
            return {'erro': str(e)}
    
    def _verificar_padrao(self, numeros: List[int], padrao: Dict) -> bool:
        """
        Verifica se os números atendem ao padrão especificado.
        """
        analise = self._analisar_numeros_sorteados(numeros)
        
        # Verificar quantidade de pares
        if 'min_pares' in padrao and analise['pares']['quantidade'] < padrao['min_pares']:
            return False
        if 'max_pares' in padrao and analise['pares']['quantidade'] > padrao['max_pares']:
            return False
        
        # Verificar sequências
        if 'min_sequencias' in padrao and len(analise['sequencias']) < padrao['min_sequencias']:
            return False
        if 'max_sequencias' in padrao and len(analise['sequencias']) > padrao['max_sequencias']:
            return False
        
        # Verificar soma
        if 'min_soma' in padrao and analise['soma_total'] < padrao['min_soma']:
            return False
        if 'max_soma' in padrao and analise['soma_total'] > padrao['max_soma']:
            return False
        
        # Verificar números específicos
        if 'contem_numeros' in padrao:
            if not all(num in numeros for num in padrao['contem_numeros']):
                return False
        
        if 'nao_contem_numeros' in padrao:
            if any(num in numeros for num in padrao['nao_contem_numeros']):
                return False
        
        return True
    
    def gerar_relatorio_visual(self, 
                              concursos: List[int], 
                              salvar_arquivo: bool = True) -> str:
        """
        Gera relatório visual com gráficos dos concursos.
        
        Args:
            concursos: Lista de números de concursos
            salvar_arquivo: Se deve salvar o arquivo
            
        Returns:
            Caminho do arquivo gerado (se salvo)
        """
        try:
            # Buscar dados dos concursos
            conn = self.conectar_db()
            
            placeholders = ','.join(['?' for _ in concursos])
            query = f"""
            SELECT numero, numeros_sorteados, data_sorteio
            FROM concursos 
            WHERE numero IN ({placeholders})
            ORDER BY numero
            """
            
            resultados = pd.read_sql_query(query, conn, params=concursos)
            conn.close()
            
            if resultados.empty:
                raise ValueError("Nenhum concurso encontrado")
            
            # Preparar dados para visualização
            todos_numeros = []
            datas = []
            
            for _, resultado in resultados.iterrows():
                numeros = json.loads(resultado['numeros_sorteados'])
                todos_numeros.extend(numeros)
                datas.append(resultado['data_sorteio'])
            
            # Criar figura com subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Análise Visual - Concursos {min(concursos)} a {max(concursos)}', fontsize=16)
            
            # Gráfico 1: Frequência de números
            frequencia = Counter(todos_numeros)
            numeros_ord = sorted(frequencia.keys())
            freq_ord = [frequencia[n] for n in numeros_ord]
            
            axes[0, 0].bar(numeros_ord, freq_ord, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Frequência dos Números')
            axes[0, 0].set_xlabel('Números')
            axes[0, 0].set_ylabel('Frequência')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Gráfico 2: Distribuição de pares/ímpares por concurso
            pares_por_concurso = []
            for _, resultado in resultados.iterrows():
                numeros = json.loads(resultado['numeros_sorteados'])
                pares = sum(1 for n in numeros if n % 2 == 0)
                pares_por_concurso.append(pares)
            
            axes[0, 1].plot(range(len(pares_por_concurso)), pares_por_concurso, 'o-', color='green')
            axes[0, 1].set_title('Números Pares por Concurso')
            axes[0, 1].set_xlabel('Concurso (sequencial)')
            axes[0, 1].set_ylabel('Quantidade de Pares')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Gráfico 3: Heatmap da cartela
            cartela = np.zeros((5, 5))
            for numero in todos_numeros:
                linha = (numero - 1) // 5
                coluna = (numero - 1) % 5
                cartela[linha, coluna] += 1
            
            sns.heatmap(cartela, annot=True, fmt='.0f', cmap='YlOrRd', 
                       ax=axes[1, 0], cbar_kws={'label': 'Frequência'})
            axes[1, 0].set_title('Mapa de Calor da Cartela')
            axes[1, 0].set_xlabel('Colunas')
            axes[1, 0].set_ylabel('Linhas')
            
            # Gráfico 4: Distribuição da soma dos números
            somas = []
            for _, resultado in resultados.iterrows():
                numeros = json.loads(resultado['numeros_sorteados'])
                somas.append(sum(numeros))
            
            axes[1, 1].hist(somas, bins=15, color='orange', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Distribuição da Soma dos Números')
            axes[1, 1].set_xlabel('Soma')
            axes[1, 1].set_ylabel('Frequência')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if salvar_arquivo:
                # Criar diretório se não existir
                Path("funcionalidades/relatorios").mkdir(parents=True, exist_ok=True)
                
                arquivo = f"funcionalidades/relatorios/relatorio_visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(arquivo, dpi=300, bbox_inches='tight')
                logger.info(f"Relatório visual salvo em: {arquivo}")
                
                plt.close()
                return arquivo
            else:
                plt.show()
                return "Gráfico exibido na tela"
                
        except Exception as e:
            logger.error(f"Erro ao gerar relatório visual: {e}")
            return f"Erro: {e}"
    
    def exportar_dados(self, 
                      concursos: List[int], 
                      formato: str = 'json',
                      incluir_analise: bool = True) -> str:
        """
        Exporta dados de concursos em diferentes formatos.
        
        Args:
            concursos: Lista de números de concursos
            formato: Formato de exportação ('json', 'csv', 'excel')
            incluir_analise: Se deve incluir análise detalhada
            
        Returns:
            Caminho do arquivo exportado
        """
        try:
            # Buscar dados
            dados_exportacao = []
            
            for concurso in concursos:
                info_concurso = self.consultar_concurso(concurso)
                if 'erro' not in info_concurso:
                    if not incluir_analise:
                        # Remover análises para exportação mais limpa
                        info_concurso.pop('analise_numeros', None)
                        info_concurso.pop('comparacao_anteriores', None)
                    
                    dados_exportacao.append(info_concurso)
            
            if not dados_exportacao:
                raise ValueError("Nenhum dado válido para exportação")
            
            # Criar diretório
            Path("funcionalidades/exportacoes").mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if formato.lower() == 'json':
                arquivo = f"funcionalidades/exportacoes/concursos_{timestamp}.json"
                with open(arquivo, 'w', encoding='utf-8') as f:
                    json.dump(dados_exportacao, f, indent=2, ensure_ascii=False)
            
            elif formato.lower() == 'csv':
                # Converter para DataFrame (formato simplificado)
                dados_simples = []
                for item in dados_exportacao:
                    dados_simples.append({
                        'concurso': item['concurso'],
                        'data_sorteio': item['data_sorteio'],
                        'numeros_sorteados': ','.join(map(str, item['numeros_sorteados'])),
                        'premio_total': item.get('premio_total', ''),
                        'ganhadores_15': item.get('ganhadores_15', ''),
                        'ganhadores_14': item.get('ganhadores_14', ''),
                        'ganhadores_13': item.get('ganhadores_13', ''),
                        'ganhadores_12': item.get('ganhadores_12', ''),
                        'ganhadores_11': item.get('ganhadores_11', '')
                    })
                
                df = pd.DataFrame(dados_simples)
                arquivo = f"funcionalidades/exportacoes/concursos_{timestamp}.csv"
                df.to_csv(arquivo, index=False, encoding='utf-8')
            
            elif formato.lower() == 'excel':
                # Similar ao CSV mas em Excel
                dados_simples = []
                for item in dados_exportacao:
                    dados_simples.append({
                        'concurso': item['concurso'],
                        'data_sorteio': item['data_sorteio'],
                        'numeros_sorteados': ','.join(map(str, item['numeros_sorteados'])),
                        'premio_total': item.get('premio_total', ''),
                        'ganhadores_15': item.get('ganhadores_15', ''),
                        'ganhadores_14': item.get('ganhadores_14', ''),
                        'ganhadores_13': item.get('ganhadores_13', ''),
                        'ganhadores_12': item.get('ganhadores_12', ''),
                        'ganhadores_11': item.get('ganhadores_11', '')
                    })
                
                df = pd.DataFrame(dados_simples)
                arquivo = f"funcionalidades/exportacoes/concursos_{timestamp}.xlsx"
                df.to_excel(arquivo, index=False)
            
            else:
                raise ValueError(f"Formato '{formato}' não suportado")
            
            logger.info(f"Dados exportados para: {arquivo}")
            return arquivo
            
        except Exception as e:
            logger.error(f"Erro na exportação: {e}")
            raise


def exemplo_uso():
    """
    Exemplo de uso da interface de sorteios.
    """
    # Inicializar interface
    interface = InterfaceSorteios()
    
    try:
        # Exemplo 1: Consultar concurso específico
        print("Consultando concurso 3000...")
        concurso = interface.consultar_concurso(3000)
        
        if 'erro' not in concurso:
            print(f"Concurso {concurso['concurso']}:")
            print(f"- Data: {concurso['data_sorteio']}")
            print(f"- Números: {concurso['numeros_sorteados']}")
            print(f"- Pares: {concurso['analise_numeros']['pares']['quantidade']}")
            print(f"- Soma: {concurso['analise_numeros']['soma_total']}")
        
        # Exemplo 2: Buscar padrões
        print("\nBuscando concursos com 8 pares...")
        padrao = {'min_pares': 8, 'max_pares': 8}
        resultados = interface.buscar_padroes(padrao, limite_resultados=5)
        
        print(f"Encontrados {resultados['total_encontrados']} concursos")
        for concurso in resultados['concursos'][:3]:
            print(f"- Concurso {concurso['concurso']}: {concurso['numeros_sorteados']}")
        
        # Exemplo 3: Gerar relatório visual
        print("\nGerando relatório visual...")
        concursos_analise = [3000, 3001, 3002, 3003, 3004]
        arquivo_relatorio = interface.gerar_relatorio_visual(concursos_analise)
        print(f"Relatório salvo em: {arquivo_relatorio}")
        
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    exemplo_uso()