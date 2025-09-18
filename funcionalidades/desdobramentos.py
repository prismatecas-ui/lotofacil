#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Desdobramentos Otimizados - Lotofácil

Este módulo implementa algoritmos avançados para desdobramentos,
incluindo otimização de combinações, redução de jogos e análise de padrões.

Autor: Sistema Lotofácil
Versão: 1.0.0
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations, product
import json
from datetime import datetime
import logging
from pathlib import Path
import math

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DesdobramentosOtimizados:
    """
    Classe para geração e otimização de desdobramentos da Lotofácil.
    
    Implementa algoritmos para:
    - Desdobramentos condicionais
    - Otimização por padrões históricos
    - Redução inteligente de jogos
    - Análise de probabilidades
    """
    
    def __init__(self, db_path: str = "database/lotofacil.db"):
        """
        Inicializa o sistema de desdobramentos.
        
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
    
    def gerar_desdobramento_condicional(self, 
                                       numeros_fixos: List[int],
                                       numeros_variaveis: List[int],
                                       condicoes: Dict = None) -> Dict:
        """
        Gera desdobramento com números fixos e variáveis.
        
        Args:
            numeros_fixos: Números que devem aparecer em todos os jogos
            numeros_variaveis: Números que podem variar entre os jogos
            condicoes: Condições específicas para o desdobramento
            
        Returns:
            Dict com o desdobramento gerado
        """
        logger.info(f"Gerando desdobramento: {len(numeros_fixos)} fixos, {len(numeros_variaveis)} variáveis")
        
        if not condicoes:
            condicoes = {
                'min_pares': 6,
                'max_pares': 10,
                'min_sequencias': 0,
                'max_sequencias': 3,
                'evitar_cantos': True
            }
        
        # Validar entrada
        if len(numeros_fixos) + len(numeros_variaveis) < self.tamanho_jogo:
            raise ValueError("Números insuficientes para formar jogos completos")
        
        if len(numeros_fixos) >= self.tamanho_jogo:
            raise ValueError("Muitos números fixos - impossível gerar variações")
        
        # Calcular quantos números variáveis precisamos
        numeros_necessarios = self.tamanho_jogo - len(numeros_fixos)
        
        # Gerar todas as combinações possíveis dos números variáveis
        combinacoes_variaveis = list(combinations(numeros_variaveis, numeros_necessarios))
        
        # Filtrar combinações baseado nas condições
        jogos_filtrados = []
        
        for comb_var in combinacoes_variaveis:
            jogo_completo = sorted(numeros_fixos + list(comb_var))
            
            if self._validar_condicoes_jogo(jogo_completo, condicoes):
                jogos_filtrados.append(tuple(jogo_completo))
        
        # Aplicar otimizações adicionais
        jogos_otimizados = self._otimizar_desdobramento(jogos_filtrados, condicoes)
        
        # Calcular estatísticas
        estatisticas = self._calcular_estatisticas_desdobramento(jogos_otimizados, condicoes)
        
        return {
            'jogos': jogos_otimizados,
            'total_jogos': len(jogos_otimizados),
            'numeros_fixos': numeros_fixos,
            'numeros_variaveis': numeros_variaveis,
            'condicoes': condicoes,
            'estatisticas': estatisticas,
            'data_geracao': datetime.now().isoformat()
        }
    
    def _validar_condicoes_jogo(self, jogo: List[int], condicoes: Dict) -> bool:
        """
        Valida se um jogo atende às condições especificadas.
        """
        # Verificar quantidade de números pares
        pares = sum(1 for num in jogo if num % 2 == 0)
        if pares < condicoes.get('min_pares', 0) or pares > condicoes.get('max_pares', 15):
            return False
        
        # Verificar sequências consecutivas
        sequencias = self._contar_sequencias(jogo)
        if sequencias < condicoes.get('min_sequencias', 0) or sequencias > condicoes.get('max_sequencias', 15):
            return False
        
        # Evitar cantos (números 1, 5, 21, 25)
        if condicoes.get('evitar_cantos', False):
            cantos = [1, 5, 21, 25]
            cantos_no_jogo = sum(1 for num in jogo if num in cantos)
            if cantos_no_jogo > 2:  # Máximo 2 cantos por jogo
                return False
        
        # Verificar distribuição por linha (cartela 5x5)
        if condicoes.get('distribuicao_equilibrada', False):
            if not self._verificar_distribuicao_equilibrada(jogo):
                return False
        
        return True
    
    def _contar_sequencias(self, jogo: List[int]) -> int:
        """
        Conta o número de sequências consecutivas no jogo.
        """
        jogo_ordenado = sorted(jogo)
        sequencias = 0
        i = 0
        
        while i < len(jogo_ordenado) - 1:
            if jogo_ordenado[i + 1] == jogo_ordenado[i] + 1:
                # Início de uma sequência
                sequencia_atual = 2
                j = i + 1
                
                while j < len(jogo_ordenado) - 1 and jogo_ordenado[j + 1] == jogo_ordenado[j] + 1:
                    sequencia_atual += 1
                    j += 1
                
                sequencias += sequencia_atual - 1  # Contar pares consecutivos
                i = j + 1
            else:
                i += 1
        
        return sequencias
    
    def _verificar_distribuicao_equilibrada(self, jogo: List[int]) -> bool:
        """
        Verifica se o jogo tem distribuição equilibrada na cartela 5x5.
        """
        # Dividir cartela em linhas (1-5, 6-10, 11-15, 16-20, 21-25)
        linhas = {
            1: [1, 2, 3, 4, 5],
            2: [6, 7, 8, 9, 10],
            3: [11, 12, 13, 14, 15],
            4: [16, 17, 18, 19, 20],
            5: [21, 22, 23, 24, 25]
        }
        
        numeros_por_linha = {}
        for linha, numeros_linha in linhas.items():
            numeros_por_linha[linha] = sum(1 for num in jogo if num in numeros_linha)
        
        # Verificar se nenhuma linha tem mais de 4 números ou menos de 1
        for linha, count in numeros_por_linha.items():
            if count > 4 or count == 0:
                return False
        
        return True
    
    def _otimizar_desdobramento(self, jogos: List[Tuple], condicoes: Dict) -> List[Tuple]:
        """
        Aplica otimizações ao desdobramento baseado em padrões históricos.
        """
        if len(jogos) <= condicoes.get('max_jogos', 100):
            return jogos
        
        # Carregar dados históricos para análise
        padroes_historicos = self._analisar_padroes_historicos()
        
        # Pontuar jogos baseado em padrões históricos
        jogos_pontuados = []
        
        for jogo in jogos:
            pontuacao = self._calcular_pontuacao_jogo(jogo, padroes_historicos)
            jogos_pontuados.append((jogo, pontuacao))
        
        # Ordenar por pontuação e selecionar os melhores
        jogos_pontuados.sort(key=lambda x: x[1], reverse=True)
        
        max_jogos = condicoes.get('max_jogos', 100)
        jogos_otimizados = [jogo for jogo, _ in jogos_pontuados[:max_jogos]]
        
        logger.info(f"Desdobramento otimizado: {len(jogos)} -> {len(jogos_otimizados)} jogos")
        
        return jogos_otimizados
    
    def _analisar_padroes_historicos(self) -> Dict:
        """
        Analisa padrões nos resultados históricos.
        """
        try:
            conn = self.conectar_db()
            
            query = """
            SELECT numeros_sorteados 
            FROM concursos 
            ORDER BY concurso DESC 
            LIMIT 200
            """
            
            resultados = pd.read_sql_query(query, conn)
            conn.close()
            
            # Analisar frequência de números
            frequencia_numeros = {i: 0 for i in range(1, 26)}
            frequencia_pares = []
            frequencia_sequencias = []
            
            for _, resultado in resultados.iterrows():
                numeros = json.loads(resultado['numeros_sorteados'])
                
                # Contar frequência de cada número
                for num in numeros:
                    frequencia_numeros[num] += 1
                
                # Contar pares
                pares = sum(1 for num in numeros if num % 2 == 0)
                frequencia_pares.append(pares)
                
                # Contar sequências
                sequencias = self._contar_sequencias(numeros)
                frequencia_sequencias.append(sequencias)
            
            return {
                'frequencia_numeros': frequencia_numeros,
                'media_pares': np.mean(frequencia_pares),
                'media_sequencias': np.mean(frequencia_sequencias),
                'total_concursos': len(resultados)
            }
            
        except Exception as e:
            logger.warning(f"Erro ao analisar padrões históricos: {e}")
            return {
                'frequencia_numeros': {i: 1 for i in range(1, 26)},
                'media_pares': 7.5,
                'media_sequencias': 2.0,
                'total_concursos': 0
            }
    
    def _calcular_pontuacao_jogo(self, jogo: Tuple, padroes: Dict) -> float:
        """
        Calcula pontuação de um jogo baseado em padrões históricos.
        """
        pontuacao = 0.0
        
        # Pontuação baseada na frequência dos números
        for num in jogo:
            freq_relativa = padroes['frequencia_numeros'].get(num, 1) / padroes['total_concursos']
            pontuacao += freq_relativa
        
        # Bonificação por proximidade à média de pares
        pares_jogo = sum(1 for num in jogo if num % 2 == 0)
        diff_pares = abs(pares_jogo - padroes['media_pares'])
        pontuacao += max(0, 5 - diff_pares)  # Bonificação inversamente proporcional à diferença
        
        # Bonificação por proximidade à média de sequências
        sequencias_jogo = self._contar_sequencias(list(jogo))
        diff_sequencias = abs(sequencias_jogo - padroes['media_sequencias'])
        pontuacao += max(0, 3 - diff_sequencias)
        
        return pontuacao
    
    def _calcular_estatisticas_desdobramento(self, jogos: List[Tuple], condicoes: Dict) -> Dict:
        """
        Calcula estatísticas detalhadas do desdobramento.
        """
        if not jogos:
            return {}
        
        # Análise de distribuição de números
        frequencia_numeros = {i: 0 for i in range(1, 26)}
        distribuicao_pares = []
        distribuicao_sequencias = []
        
        for jogo in jogos:
            # Contar frequência de cada número
            for num in jogo:
                frequencia_numeros[num] += 1
            
            # Analisar pares
            pares = sum(1 for num in jogo if num % 2 == 0)
            distribuicao_pares.append(pares)
            
            # Analisar sequências
            sequencias = self._contar_sequencias(list(jogo))
            distribuicao_sequencias.append(sequencias)
        
        # Calcular cobertura de números
        numeros_cobertos = len([num for num, freq in frequencia_numeros.items() if freq > 0])
        cobertura_percentual = (numeros_cobertos / 25) * 100
        
        return {
            'total_jogos': len(jogos),
            'numeros_cobertos': numeros_cobertos,
            'cobertura_percentual': round(cobertura_percentual, 2),
            'frequencia_numeros': frequencia_numeros,
            'media_pares': round(np.mean(distribuicao_pares), 2),
            'desvio_pares': round(np.std(distribuicao_pares), 2),
            'media_sequencias': round(np.mean(distribuicao_sequencias), 2),
            'desvio_sequencias': round(np.std(distribuicao_sequencias), 2),
            'distribuicao_pares': {
                str(i): distribuicao_pares.count(i) for i in range(0, 16)
            },
            'custo_estimado': len(jogos) * 2.50  # Valor padrão da aposta
        }
    
    def gerar_desdobramento_inteligente(self, 
                                       orcamento: float = 100.0,
                                       estrategia: str = 'equilibrada') -> Dict:
        """
        Gera desdobramento inteligente baseado em orçamento e estratégia.
        
        Args:
            orcamento: Orçamento disponível em reais
            estrategia: Tipo de estratégia ('conservadora', 'equilibrada', 'agressiva')
            
        Returns:
            Dict com desdobramento otimizado
        """
        valor_aposta = 2.50
        max_jogos = int(orcamento / valor_aposta)
        
        # Definir parâmetros baseado na estratégia
        estrategias = {
            'conservadora': {
                'numeros_fixos_min': 8,
                'numeros_fixos_max': 12,
                'condicoes': {
                    'min_pares': 6, 'max_pares': 9,
                    'min_sequencias': 1, 'max_sequencias': 3,
                    'evitar_cantos': True,
                    'distribuicao_equilibrada': True
                }
            },
            'equilibrada': {
                'numeros_fixos_min': 5,
                'numeros_fixos_max': 10,
                'condicoes': {
                    'min_pares': 5, 'max_pares': 10,
                    'min_sequencias': 0, 'max_sequencias': 4,
                    'evitar_cantos': False,
                    'distribuicao_equilibrada': True
                }
            },
            'agressiva': {
                'numeros_fixos_min': 3,
                'numeros_fixos_max': 8,
                'condicoes': {
                    'min_pares': 4, 'max_pares': 11,
                    'min_sequencias': 0, 'max_sequencias': 5,
                    'evitar_cantos': False,
                    'distribuicao_equilibrada': False
                }
            }
        }
        
        params = estrategias.get(estrategia, estrategias['equilibrada'])
        
        # Selecionar números baseado em análise histórica
        padroes = self._analisar_padroes_historicos()
        numeros_selecionados = self._selecionar_numeros_inteligente(padroes, estrategia)
        
        # Dividir em fixos e variáveis
        num_fixos = min(params['numeros_fixos_max'], len(numeros_selecionados) // 2)
        numeros_fixos = numeros_selecionados[:num_fixos]
        numeros_variaveis = numeros_selecionados[num_fixos:]
        
        # Adicionar condição de máximo de jogos
        params['condicoes']['max_jogos'] = max_jogos
        
        # Gerar desdobramento
        desdobramento = self.gerar_desdobramento_condicional(
            numeros_fixos,
            numeros_variaveis,
            params['condicoes']
        )
        
        # Adicionar informações da estratégia
        desdobramento['estrategia'] = estrategia
        desdobramento['orcamento_usado'] = desdobramento['total_jogos'] * valor_aposta
        desdobramento['orcamento_restante'] = orcamento - desdobramento['orcamento_usado']
        
        return desdobramento
    
    def _selecionar_numeros_inteligente(self, padroes: Dict, estrategia: str) -> List[int]:
        """
        Seleciona números de forma inteligente baseado em padrões e estratégia.
        """
        # Ordenar números por frequência
        numeros_ordenados = sorted(
            padroes['frequencia_numeros'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if estrategia == 'conservadora':
            # Preferir números mais frequentes
            return [num for num, _ in numeros_ordenados[:18]]
        elif estrategia == 'agressiva':
            # Misturar números frequentes e menos frequentes
            frequentes = [num for num, _ in numeros_ordenados[:12]]
            menos_frequentes = [num for num, _ in numeros_ordenados[-8:]]
            return frequentes + menos_frequentes
        else:  # equilibrada
            # Selecionar números do meio da distribuição
            return [num for num, _ in numeros_ordenados[3:21]]
    
    def salvar_desdobramento(self, desdobramento: Dict, nome: str) -> str:
        """
        Salva desdobramento em arquivo JSON.
        
        Args:
            desdobramento: Dados do desdobramento
            nome: Nome do arquivo
            
        Returns:
            Caminho do arquivo salvo
        """
        # Criar diretório se não existir
        Path("funcionalidades/desdobramentos").mkdir(parents=True, exist_ok=True)
        
        arquivo = f"funcionalidades/desdobramentos/{nome}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(arquivo, 'w', encoding='utf-8') as f:
            json.dump(desdobramento, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Desdobramento salvo em: {arquivo}")
        return arquivo


def exemplo_uso():
    """
    Exemplo de uso do sistema de desdobramentos.
    """
    # Inicializar sistema
    desdobramentos = DesdobramentosOtimizados()
    
    try:
        # Exemplo 1: Desdobramento condicional
        print("Gerando desdobramento condicional...")
        numeros_fixos = [7, 14, 21]  # Números que devem aparecer em todos os jogos
        numeros_variaveis = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]
        
        desdobramento = desdobramentos.gerar_desdobramento_condicional(
            numeros_fixos,
            numeros_variaveis,
            {'max_jogos': 30, 'min_pares': 6, 'max_pares': 9}
        )
        
        print(f"Desdobramento gerado:")
        print(f"- Total de jogos: {desdobramento['total_jogos']}")
        print(f"- Cobertura: {desdobramento['estatisticas']['cobertura_percentual']}%")
        print(f"- Custo estimado: R$ {desdobramento['estatisticas']['custo_estimado']:.2f}")
        
        # Exemplo 2: Desdobramento inteligente
        print("\nGerando desdobramento inteligente...")
        inteligente = desdobramentos.gerar_desdobramento_inteligente(
            orcamento=75.0,
            estrategia='equilibrada'
        )
        
        print(f"Desdobramento inteligente:")
        print(f"- Estratégia: {inteligente['estrategia']}")
        print(f"- Total de jogos: {inteligente['total_jogos']}")
        print(f"- Orçamento usado: R$ {inteligente['orcamento_usado']:.2f}")
        print(f"- Orçamento restante: R$ {inteligente['orcamento_restante']:.2f}")
        
        # Salvar desdobramento
        arquivo = desdobramentos.salvar_desdobramento(inteligente, "exemplo_inteligente")
        print(f"\nDesdobramento salvo em: {arquivo}")
        
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    exemplo_uso()