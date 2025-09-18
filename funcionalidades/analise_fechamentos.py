#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Análise de Fechamentos - Lotofácil

Este módulo implementa algoritmos avançados para análise de fechamentos,
incluindo cálculos de cobertura, otimização de apostas e análise de garantias.

Autor: Sistema Lotofácil
Versão: 1.0.0
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations
import json
from datetime import datetime
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnaliseFechamentos:
    """
    Classe principal para análise de fechamentos da Lotofácil.
    
    Implementa algoritmos para:
    - Cálculo de fechamentos com garantias
    - Otimização de apostas
    - Análise de cobertura
    - Redução de jogos
    """
    
    def __init__(self, db_path: str = "database/lotofacil.db"):
        """
        Inicializa o sistema de análise de fechamentos.
        
        Args:
            db_path: Caminho para o banco de dados SQLite
        """
        self.db_path = db_path
        self.numeros_lotofacil = list(range(1, 26))  # 1 a 25
        self.tamanho_jogo = 15  # Lotofácil tem 15 números por jogo
        
    def conectar_db(self) -> sqlite3.Connection:
        """Estabelece conexão com o banco de dados."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            logger.error(f"Erro ao conectar com o banco: {e}")
            raise
    
    def calcular_fechamento_garantia(self, 
                                   numeros_escolhidos: List[int], 
                                   garantia: int = 11,
                                   max_jogos: int = 100) -> Dict:
        """
        Calcula fechamento com garantia específica.
        
        Args:
            numeros_escolhidos: Lista de números escolhidos (16-20 números)
            garantia: Número mínimo de acertos garantidos (padrão: 11)
            max_jogos: Número máximo de jogos no fechamento
            
        Returns:
            Dict com informações do fechamento
        """
        logger.info(f"Calculando fechamento para {len(numeros_escolhidos)} números")
        logger.info(f"Garantia: {garantia} acertos, Máximo: {max_jogos} jogos")
        
        if len(numeros_escolhidos) < 16 or len(numeros_escolhidos) > 20:
            raise ValueError("Número de dezenas deve estar entre 16 e 20")
        
        # Gerar todas as combinações possíveis de 15 números
        todas_combinacoes = list(combinations(numeros_escolhidos, self.tamanho_jogo))
        
        # Aplicar algoritmo de cobertura para encontrar fechamento ótimo
        fechamento_otimo = self._algoritmo_cobertura(
            todas_combinacoes, 
            numeros_escolhidos, 
            garantia, 
            max_jogos
        )
        
        # Calcular estatísticas do fechamento
        estatisticas = self._calcular_estatisticas_fechamento(
            fechamento_otimo, 
            numeros_escolhidos, 
            garantia
        )
        
        return {
            'jogos': fechamento_otimo,
            'total_jogos': len(fechamento_otimo),
            'numeros_base': numeros_escolhidos,
            'garantia': garantia,
            'estatisticas': estatisticas,
            'data_calculo': datetime.now().isoformat()
        }
    
    def _algoritmo_cobertura(self, 
                           combinacoes: List[Tuple], 
                           numeros_base: List[int], 
                           garantia: int, 
                           max_jogos: int) -> List[Tuple]:
        """
        Implementa algoritmo de cobertura para encontrar fechamento ótimo.
        
        Utiliza heurística gulosa para selecionar jogos que maximizam
        a cobertura com o menor número de apostas.
        """
        fechamento = []
        combinacoes_cobertas = set()
        
        # Gerar todas as combinações de 'garantia' números dos números base
        combinacoes_garantia = list(combinations(numeros_base, garantia))
        
        while len(fechamento) < max_jogos and len(combinacoes_cobertas) < len(combinacoes_garantia):
            melhor_jogo = None
            melhor_cobertura = 0
            
            # Encontrar o jogo que cobre mais combinações não cobertas
            for jogo in combinacoes:
                if jogo in fechamento:
                    continue
                    
                cobertura_atual = 0
                for comb_garantia in combinacoes_garantia:
                    if comb_garantia not in combinacoes_cobertas:
                        # Verificar se o jogo cobre esta combinação
                        if set(comb_garantia).issubset(set(jogo)):
                            cobertura_atual += 1
                
                if cobertura_atual > melhor_cobertura:
                    melhor_cobertura = cobertura_atual
                    melhor_jogo = jogo
            
            if melhor_jogo is None:
                break
                
            fechamento.append(melhor_jogo)
            
            # Atualizar combinações cobertas
            for comb_garantia in combinacoes_garantia:
                if set(comb_garantia).issubset(set(melhor_jogo)):
                    combinacoes_cobertas.add(comb_garantia)
        
        return fechamento
    
    def _calcular_estatisticas_fechamento(self, 
                                        fechamento: List[Tuple], 
                                        numeros_base: List[int], 
                                        garantia: int) -> Dict:
        """
        Calcula estatísticas detalhadas do fechamento.
        """
        total_combinacoes_garantia = len(list(combinations(numeros_base, garantia)))
        combinacoes_cobertas = set()
        
        # Calcular cobertura real
        for jogo in fechamento:
            for comb_garantia in combinations(numeros_base, garantia):
                if set(comb_garantia).issubset(set(jogo)):
                    combinacoes_cobertas.add(comb_garantia)
        
        cobertura_percentual = (len(combinacoes_cobertas) / total_combinacoes_garantia) * 100
        
        # Calcular custo por combinação coberta
        custo_por_combinacao = len(fechamento) / len(combinacoes_cobertas) if combinacoes_cobertas else 0
        
        # Analisar distribuição de números
        frequencia_numeros = {}
        for numero in numeros_base:
            frequencia_numeros[numero] = sum(1 for jogo in fechamento if numero in jogo)
        
        return {
            'total_combinacoes_garantia': total_combinacoes_garantia,
            'combinacoes_cobertas': len(combinacoes_cobertas),
            'cobertura_percentual': round(cobertura_percentual, 2),
            'custo_por_combinacao': round(custo_por_combinacao, 2),
            'frequencia_numeros': frequencia_numeros,
            'eficiencia': round(len(combinacoes_cobertas) / len(fechamento), 2)
        }
    
    def analisar_fechamento_historico(self, fechamento: List[Tuple]) -> Dict:
        """
        Analisa o desempenho de um fechamento contra resultados históricos.
        
        Args:
            fechamento: Lista de jogos do fechamento
            
        Returns:
            Dict com análise de desempenho
        """
        try:
            conn = self.conectar_db()
            
            # Buscar resultados históricos
            query = """
            SELECT numero as concurso, dezenas as numeros_sorteados 
            FROM concursos 
            ORDER BY numero DESC 
            LIMIT 100
            """
            
            resultados = pd.read_sql_query(query, conn)
            conn.close()
            
            acertos_por_concurso = []
            
            for _, resultado in resultados.iterrows():
                numeros_sorteados = json.loads(resultado['numeros_sorteados'])
                
                melhor_acerto = 0
                for jogo in fechamento:
                    acertos = len(set(jogo).intersection(set(numeros_sorteados)))
                    melhor_acerto = max(melhor_acerto, acertos)
                
                acertos_por_concurso.append({
                    'concurso': resultado['concurso'],
                    'melhor_acerto': melhor_acerto
                })
            
            # Calcular estatísticas de desempenho
            acertos = [item['melhor_acerto'] for item in acertos_por_concurso]
            
            return {
                'total_concursos_analisados': len(acertos_por_concurso),
                'acerto_medio': round(np.mean(acertos), 2),
                'acerto_maximo': max(acertos),
                'acerto_minimo': min(acertos),
                'distribuicao_acertos': {
                    str(i): acertos.count(i) for i in range(11, 16)
                },
                'percentual_11_ou_mais': round((sum(1 for a in acertos if a >= 11) / len(acertos)) * 100, 2),
                'percentual_12_ou_mais': round((sum(1 for a in acertos if a >= 12) / len(acertos)) * 100, 2),
                'percentual_13_ou_mais': round((sum(1 for a in acertos if a >= 13) / len(acertos)) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise histórica: {e}")
            return {}
    
    def otimizar_fechamento_custo_beneficio(self, 
                                          numeros_escolhidos: List[int], 
                                          orcamento_maximo: float = 100.0,
                                          valor_aposta: float = 2.50) -> Dict:
        """
        Otimiza fechamento considerando custo-benefício.
        
        Args:
            numeros_escolhidos: Lista de números escolhidos
            orcamento_maximo: Orçamento máximo em reais
            valor_aposta: Valor de cada aposta
            
        Returns:
            Dict com fechamento otimizado
        """
        max_jogos = int(orcamento_maximo / valor_aposta)
        
        # Testar diferentes garantias para encontrar a melhor relação custo-benefício
        melhores_opcoes = []
        
        for garantia in range(11, 14):  # Testar garantias 11, 12 e 13
            try:
                fechamento = self.calcular_fechamento_garantia(
                    numeros_escolhidos, 
                    garantia, 
                    max_jogos
                )
                
                custo_total = fechamento['total_jogos'] * valor_aposta
                eficiencia = fechamento['estatisticas']['eficiencia']
                
                if custo_total <= orcamento_maximo:
                    melhores_opcoes.append({
                        'garantia': garantia,
                        'fechamento': fechamento,
                        'custo_total': custo_total,
                        'eficiencia': eficiencia,
                        'custo_beneficio': eficiencia / custo_total
                    })
                    
            except Exception as e:
                logger.warning(f"Erro ao calcular garantia {garantia}: {e}")
                continue
        
        if not melhores_opcoes:
            raise ValueError("Não foi possível calcular fechamento dentro do orçamento")
        
        # Selecionar a melhor opção (maior custo-benefício)
        melhor_opcao = max(melhores_opcoes, key=lambda x: x['custo_beneficio'])
        
        return {
            'fechamento_otimizado': melhor_opcao['fechamento'],
            'custo_total': melhor_opcao['custo_total'],
            'jogos_restantes': max_jogos - melhor_opcao['fechamento']['total_jogos'],
            'orcamento_restante': orcamento_maximo - melhor_opcao['custo_total'],
            'todas_opcoes': melhores_opcoes
        }
    
    def salvar_fechamento(self, fechamento: Dict, nome: str) -> str:
        """
        Salva fechamento em arquivo JSON.
        
        Args:
            fechamento: Dados do fechamento
            nome: Nome do arquivo
            
        Returns:
            Caminho do arquivo salvo
        """
        # Criar diretório se não existir
        Path("funcionalidades/fechamentos").mkdir(parents=True, exist_ok=True)
        
        arquivo = f"funcionalidades/fechamentos/{nome}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(arquivo, 'w', encoding='utf-8') as f:
            json.dump(fechamento, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Fechamento salvo em: {arquivo}")
        return arquivo
    
    def carregar_fechamento(self, arquivo: str) -> Dict:
        """
        Carrega fechamento de arquivo JSON.
        
        Args:
            arquivo: Caminho do arquivo
            
        Returns:
            Dados do fechamento
        """
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar fechamento: {e}")
            raise
    
    def analisar_padroes(self, dados_historicos):
        """
        Analisa padrões nos dados históricos dos concursos.
        
        Args:
            dados_historicos (DataFrame): DataFrame com dados dos concursos
            
        Returns:
            dict: Dicionário com análises de padrões
        """
        try:
            padroes = {
                'frequencias': {},
                'sequencias': [],
                'paridade': {'pares': 0, 'impares': 0}
            }
            
            # Análise de frequência
            for _, row in dados_historicos.iterrows():
                # Verificar se existe coluna 'dezenas' ou 'numeros_sorteados'
                numeros = []
                if 'dezenas' in row and row['dezenas'] and row['dezenas'].strip():
                    # Se dezenas é uma string com números separados por vírgula
                    if ',' in str(row['dezenas']):
                        numeros = [int(x.strip()) for x in str(row['dezenas']).split(',') if x.strip().isdigit()]
                    else:
                        # Tentar como JSON
                        try:
                            numeros = json.loads(row['dezenas'])
                        except:
                            continue
                elif 'numeros_sorteados' in row and row['numeros_sorteados']:
                    try:
                        numeros = json.loads(row['numeros_sorteados'])
                    except:
                        continue
                else:
                    # Tentar colunas individuais n01, n02, etc.
                    numeros = []
                    for i in range(1, 26):
                        col_name = f'n{i:02d}'
                        if col_name in row and row[col_name] == 1:
                            numeros.append(i)
                
                if not numeros:
                    continue
                    
                for numero in numeros:
                    if numero in padroes['frequencias']:
                        padroes['frequencias'][numero] += 1
                    else:
                        padroes['frequencias'][numero] = 1
                    
                    # Análise de paridade
                    if numero % 2 == 0:
                        padroes['paridade']['pares'] += 1
                    else:
                        padroes['paridade']['impares'] += 1
            
            # Análise de sequências (números consecutivos)
            for _, row in dados_historicos.iterrows():
                # Obter números da mesma forma que acima
                numeros = []
                if 'dezenas' in row and row['dezenas'] and row['dezenas'].strip():
                    if ',' in str(row['dezenas']):
                        numeros = [int(x.strip()) for x in str(row['dezenas']).split(',') if x.strip().isdigit()]
                    else:
                        try:
                            numeros = json.loads(row['dezenas'])
                        except:
                            continue
                elif 'numeros_sorteados' in row and row['numeros_sorteados']:
                    try:
                        numeros = json.loads(row['numeros_sorteados'])
                    except:
                        continue
                else:
                    numeros = []
                    for i in range(1, 26):
                        col_name = f'n{i:02d}'
                        if col_name in row and row[col_name] == 1:
                            numeros.append(i)
                
                if not numeros:
                    continue
                    
                numeros = sorted(numeros)
                sequencia_atual = []
                
                for i in range(len(numeros) - 1):
                    if numeros[i+1] - numeros[i] == 1:
                        if not sequencia_atual:
                            sequencia_atual = [numeros[i], numeros[i+1]]
                        else:
                            sequencia_atual.append(numeros[i+1])
                    else:
                        if len(sequencia_atual) >= 2:
                            padroes['sequencias'].append(sequencia_atual)
                        sequencia_atual = []
                
                if len(sequencia_atual) >= 2:
                    padroes['sequencias'].append(sequencia_atual)
            
            return padroes
            
        except Exception as e:
            logger.error(f"Erro na análise de padrões: {e}")
            return {'frequencias': {}, 'sequencias': [], 'paridade': {'pares': 0, 'impares': 0}}


def exemplo_uso():
    """
    Exemplo de uso do sistema de análise de fechamentos.
    """
    # Inicializar sistema
    analise = AnaliseFechamentos()
    
    # Números escolhidos (exemplo: 18 números)
    numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    try:
        # Calcular fechamento com garantia 11
        print("Calculando fechamento com garantia 11...")
        fechamento = analise.calcular_fechamento_garantia(numeros, garantia=11, max_jogos=50)
        
        print(f"Fechamento calculado:")
        print(f"- Total de jogos: {fechamento['total_jogos']}")
        print(f"- Cobertura: {fechamento['estatisticas']['cobertura_percentual']}%")
        print(f"- Eficiência: {fechamento['estatisticas']['eficiencia']}")
        
        # Otimizar por custo-benefício
        print("\nOtimizando por custo-benefício...")
        otimizado = analise.otimizar_fechamento_custo_beneficio(numeros, orcamento_maximo=125.0)
        
        print(f"Fechamento otimizado:")
        print(f"- Custo total: R$ {otimizado['custo_total']:.2f}")
        print(f"- Orçamento restante: R$ {otimizado['orcamento_restante']:.2f}")
        
        # Salvar fechamento
        arquivo = analise.salvar_fechamento(fechamento, "exemplo_fechamento")
        print(f"\nFechamento salvo em: {arquivo}")
        
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    exemplo_uso()