#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Feature Engineering Avançado para Sistema Lotofácil
Implementa features estatísticas, temporais e de padrões para melhorar predições
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineeringLotofacil:
    """
    Classe principal para feature engineering avançado da Lotofácil
    """
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(n_quantiles=100, random_state=42)
        }
        self.feature_selector = None
        self.feature_names = []
        
    def criar_features_estatisticas(self, dados_historicos: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features estatísticas avançadas para cada concurso
        
        Args:
            dados_historicos: DataFrame com histórico de concursos
            
        Returns:
            DataFrame com features estatísticas
        """
        logger.info("Criando features estatísticas avançadas...")
        
        features = []
        concursos_processados = 0
        
        for idx, concurso in dados_historicos.iterrows():
            try:
                # Extrair números do concurso usando o método atualizado
                numeros = self._extrair_numeros_concurso(concurso)
                
                if not numeros or len(numeros) != 15:
                    logger.warning(f"Concurso {concurso.get('Concurso', idx)}: números inválidos ({len(numeros) if numeros else 0} números), pulando...")
                    continue
                
                # Validar se todos os números estão no range válido (1-25)
                if not all(1 <= n <= 25 for n in numeros):
                    logger.warning(f"Concurso {concurso.get('Concurso', idx)}: números fora do range válido, pulando...")
                    continue
                    
                # Calcular features estatísticas
                feature_vector = {}
                
                try:
                    feature_vector.update(self._calcular_features_basicas(numeros))
                except Exception as e:
                    logger.warning(f"Erro ao calcular features básicas para concurso {concurso.get('Concurso', idx)}: {e}")
                    continue
                
                try:
                    feature_vector.update(self._calcular_features_distribuicao(numeros))
                except Exception as e:
                    logger.warning(f"Erro ao calcular features de distribuição para concurso {concurso.get('Concurso', idx)}: {e}")
                    continue
                
                try:
                    feature_vector.update(self._calcular_features_sequencias(numeros))
                except Exception as e:
                    logger.warning(f"Erro ao calcular features de sequências para concurso {concurso.get('Concurso', idx)}: {e}")
                    continue
                
                try:
                    feature_vector.update(self._calcular_features_matematicas(numeros))
                except Exception as e:
                    logger.warning(f"Erro ao calcular features matemáticas para concurso {concurso.get('Concurso', idx)}: {e}")
                    continue
                
                try:
                    feature_vector.update(self._calcular_features_posicionais(numeros))
                except Exception as e:
                    logger.warning(f"Erro ao calcular features posicionais para concurso {concurso.get('Concurso', idx)}: {e}")
                    continue
                
                # Adicionar informações do concurso
                feature_vector['concurso'] = concurso.get('Concurso', idx)
                
                features.append(feature_vector)
                concursos_processados += 1
                
            except Exception as e:
                logger.error(f"Erro geral ao processar concurso {concurso.get('Concurso', idx)}: {e}")
                continue
        
        if not features:
            logger.error("Nenhuma feature estatística foi criada!")
            return pd.DataFrame()
        
        df_features = pd.DataFrame(features)
        logger.info(f"Features estatísticas criadas: {len(df_features.columns)} colunas para {concursos_processados} concursos")
        
        return df_features
    
    def _extrair_numeros_concurso(self, concurso: pd.Series) -> List[int]:
        """
        Extrai números do concurso de diferentes formatos possíveis
        """
        try:
            # Primeiro tentar colunas B1-B15 (formato principal)
            numeros = []
            for i in range(1, 16):
                col_name = f'B{i}'
                if col_name in concurso and pd.notna(concurso[col_name]):
                    valor = concurso[col_name]
                    if isinstance(valor, (int, float)) and not pd.isna(valor):
                        numeros.append(int(valor))
                    elif isinstance(valor, str) and valor.strip().isdigit():
                        numeros.append(int(valor.strip()))
            
            if len(numeros) == 15:
                return numeros
            
            # Tentar diferentes colunas possíveis como fallback
            for col in ['dezenas', 'numeros', 'bolas', 'resultado']:
                if col in concurso and pd.notna(concurso[col]):
                    valor = concurso[col]
                    if isinstance(valor, str):
                        # Tentar diferentes separadores
                        for sep in [',', '-', ' ', ';']:
                            if sep in valor:
                                try:
                                    nums = [int(x.strip()) for x in valor.split(sep) if x.strip().isdigit()]
                                    if len(nums) == 15:
                                        return nums
                                except:
                                    continue
                    elif isinstance(valor, list) and len(valor) == 15:
                        return [int(x) for x in valor if str(x).isdigit()]
            
            # Tentar colunas individuais alternativas
            numeros = []
            for i in range(1, 16):
                for col_pattern in [f'bola_{i:02d}', f'bola_{i}', f'n{i}', f'num_{i}']:
                    if col_pattern in concurso and pd.notna(concurso[col_pattern]):
                        numeros.append(int(concurso[col_pattern]))
                        break
            
            return numeros if len(numeros) == 15 else []
            
        except Exception as e:
            logger.warning(f"Erro ao extrair números do concurso: {e}")
            return []
    
    def _calcular_features_basicas(self, numeros: List[int]) -> Dict:
        """
        Calcula features básicas de distribuição
        """
        return {
            # Distribuição par/ímpar
            'pares': sum(1 for n in numeros if n % 2 == 0),
            'impares': sum(1 for n in numeros if n % 2 == 1),
            'razao_par_impar': sum(1 for n in numeros if n % 2 == 0) / len(numeros),
            
            # Distribuição por faixas
            'baixos': sum(1 for n in numeros if n <= 8),
            'medios': sum(1 for n in numeros if 9 <= n <= 17),
            'altos': sum(1 for n in numeros if n >= 18),
            
            # Distribuição detalhada por faixas
            'faixa_1_5': sum(1 for n in numeros if 1 <= n <= 5),
            'faixa_6_10': sum(1 for n in numeros if 6 <= n <= 10),
            'faixa_11_15': sum(1 for n in numeros if 11 <= n <= 15),
            'faixa_16_20': sum(1 for n in numeros if 16 <= n <= 20),
            'faixa_21_25': sum(1 for n in numeros if 21 <= n <= 25),
        }
    
    def _calcular_features_distribuicao(self, numeros: List[int]) -> Dict:
        """
        Calcula features de distribuição por colunas da cartela
        """
        # Distribuição por colunas da cartela Lotofácil
        colunas = {
            'coluna_1': [1, 6, 11, 16, 21],
            'coluna_2': [2, 7, 12, 17, 22],
            'coluna_3': [3, 8, 13, 18, 23],
            'coluna_4': [4, 9, 14, 19, 24],
            'coluna_5': [5, 10, 15, 20, 25]
        }
        
        features = {}
        for nome_coluna, nums_coluna in colunas.items():
            features[nome_coluna] = sum(1 for n in numeros if n in nums_coluna)
        
        # Distribuição por linhas
        linhas = {
            'linha_1': list(range(1, 6)),
            'linha_2': list(range(6, 11)),
            'linha_3': list(range(11, 16)),
            'linha_4': list(range(16, 21)),
            'linha_5': list(range(21, 26))
        }
        
        for nome_linha, nums_linha in linhas.items():
            features[nome_linha] = sum(1 for n in numeros if n in nums_linha)
        
        # Distribuição por quadrantes
        features['quadrante_1'] = sum(1 for n in numeros if n in [1,2,3,6,7,8,11,12,13])
        features['quadrante_2'] = sum(1 for n in numeros if n in [3,4,5,8,9,10,13,14,15])
        features['quadrante_3'] = sum(1 for n in numeros if n in [11,12,13,16,17,18,21,22,23])
        features['quadrante_4'] = sum(1 for n in numeros if n in [13,14,15,18,19,20,23,24,25])
        
        return features
    
    def _calcular_features_sequencias(self, numeros: List[int]) -> Dict:
        """
        Calcula features relacionadas a sequências e padrões
        """
        if not numeros:
            return {}
        
        try:
            numeros_sorted = sorted(numeros)
            
            # Contar sequências consecutivas
            consecutivos = self._contar_consecutivos(numeros_sorted)
            
            # Calcular gaps entre números
            gaps = [numeros_sorted[i] - numeros_sorted[i-1] for i in range(1, len(numeros_sorted))]
            
            # Calcular amplitude e densidade com proteção contra divisão por zero
            amplitude = max(numeros) - min(numeros) if numeros else 0
            densidade = 0
            if numeros and amplitude > 0:
                densidade = len(set(numeros)) / (amplitude + 1)
            
            return {
                'consecutivos_max': consecutivos,
                'total_sequencias': self._contar_total_sequencias(numeros_sorted),
                'gaps_medio': float(np.mean(gaps)) if gaps else 0.0,
                'gaps_std': float(np.std(gaps)) if gaps else 0.0,
                'gaps_min': min(gaps) if gaps else 0,
                'gaps_max': max(gaps) if gaps else 0,
                'amplitude': amplitude,
                'densidade': float(densidade)
            }
        except Exception as e:
            logger.warning(f"Erro ao calcular features de sequências: {e}")
            return {
                'consecutivos_max': 0,
                'total_sequencias': 0,
                'gaps_medio': 0.0,
                'gaps_std': 0.0,
                'gaps_min': 0,
                'gaps_max': 0,
                'amplitude': 0,
                'densidade': 0.0
            }
    
    def _calcular_features_matematicas(self, numeros: List[int]) -> Dict:
        """
        Calcula features matemáticas e estatísticas
        """
        if not numeros:
            return {}
        
        try:
            media = np.mean(numeros)
            desvio = np.std(numeros)
            
            return {
                'soma_total': sum(numeros),
                'media': float(media),
                'mediana': float(np.median(numeros)),
                'desvio_padrao': float(desvio),
                'variancia': float(np.var(numeros)),
                'coef_variacao': float(desvio / media) if media != 0 else 0.0,
                'assimetria': self._calcular_assimetria(numeros),
                'curtose': self._calcular_curtose(numeros),
                'entropia': self._calcular_entropia(numeros)
            }
        except Exception as e:
            logger.warning(f"Erro ao calcular features matemáticas: {e}")
            return {
                'soma_total': sum(numeros) if numeros else 0,
                'media': 0.0,
                'mediana': 0.0,
                'desvio_padrao': 0.0,
                'variancia': 0.0,
                'coef_variacao': 0.0,
                'assimetria': 0.0,
                'curtose': 0.0,
                'entropia': 0.0
            }
    
    def _calcular_features_posicionais(self, numeros: List[int]) -> Dict:
        """
        Calcula features baseadas em posições específicas
        """
        numeros_sorted = sorted(numeros)
        
        return {
            'primeiro_numero': numeros_sorted[0],
            'ultimo_numero': numeros_sorted[-1],
            'numero_central': numeros_sorted[7],  # 8º número (meio)
            'quartil_1': numeros_sorted[3],       # 4º número
            'quartil_3': numeros_sorted[11],      # 12º número
            'numeros_extremos': sum(1 for n in numeros if n in [1,2,24,25]),
            'numeros_centrais': sum(1 for n in numeros if 10 <= n <= 16)
        }
    
    def _contar_consecutivos(self, numeros_sorted: List[int]) -> int:
        """
        Conta a maior sequência de números consecutivos
        """
        if not numeros_sorted:
            return 0
            
        max_consecutivos = 1
        atual = 1
        
        for i in range(1, len(numeros_sorted)):
            if numeros_sorted[i] == numeros_sorted[i-1] + 1:
                atual += 1
                max_consecutivos = max(max_consecutivos, atual)
            else:
                atual = 1
        
        return max_consecutivos
    
    def _contar_total_sequencias(self, numeros_sorted: List[int]) -> int:
        """
        Conta o total de sequências (grupos de 2+ números consecutivos)
        """
        if not numeros_sorted:
            return 0
            
        sequencias = 0
        em_sequencia = False
        
        for i in range(1, len(numeros_sorted)):
            if numeros_sorted[i] == numeros_sorted[i-1] + 1:
                if not em_sequencia:
                    sequencias += 1
                    em_sequencia = True
            else:
                em_sequencia = False
        
        return sequencias
    
    def _calcular_assimetria(self, numeros: List[int]) -> float:
        """
        Calcula assimetria da distribuição
        """
        if len(numeros) < 3:
            return 0
        
        media = np.mean(numeros)
        std = np.std(numeros)
        
        if std == 0:
            return 0
        
        n = len(numeros)
        assimetria = (n / ((n-1) * (n-2))) * sum(((x - media) / std) ** 3 for x in numeros)
        
        return assimetria
    
    def _calcular_curtose(self, numeros: List[int]) -> float:
        """
        Calcula curtose da distribuição
        """
        if len(numeros) < 4:
            return 0
        
        media = np.mean(numeros)
        std = np.std(numeros)
        
        if std == 0:
            return 0
        
        n = len(numeros)
        curtose = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * sum(((x - media) / std) ** 4 for x in numeros)
        curtose -= 3 * (n-1) ** 2 / ((n-2) * (n-3))
        
        return curtose
    
    def _calcular_entropia(self, numeros: List[int]) -> float:
        """
        Calcula entropia da distribuição dos números
        """
        # Dividir em bins para calcular entropia
        bins = np.histogram(numeros, bins=5, range=(1, 25))[0]
        bins = bins[bins > 0]  # Remover bins vazios
        
        if len(bins) == 0:
            return 0
        
        # Normalizar
        probs = bins / sum(bins)
        
        # Calcular entropia
        entropia = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropia
    
    def criar_features_temporais(self, dados_historicos: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features temporais baseadas em padrões históricos - OTIMIZADO
        
        Args:
            dados_historicos: DataFrame com histórico ordenado por data
            
        Returns:
            DataFrame com features temporais
        """
        logger.info("Criando features temporais...")
        
        features_temporais = []
        total_concursos = len(dados_historicos)
        
        # Processar em lotes para melhor performance
        batch_size = 100
        
        for batch_start in range(0, total_concursos, batch_size):
            batch_end = min(batch_start + batch_size, total_concursos)
            
            # Log de progresso
            if batch_start % 500 == 0:
                logger.info(f"Processando concursos {batch_start+1} a {batch_end} de {total_concursos}...")
        
            for i in range(batch_start, batch_end):
                concurso = dados_historicos.iloc[i]
                # Extrair números do concurso atual
                numeros_atual = self._extrair_numeros_concurso(concurso)
                
                if not numeros_atual:
                    continue
                
                # Análise dos últimos N concursos
                ultimos_5 = dados_historicos.iloc[max(0, i-5):i] if i > 0 else pd.DataFrame()
                ultimos_10 = dados_historicos.iloc[max(0, i-10):i] if i > 0 else pd.DataFrame()
                ultimos_20 = dados_historicos.iloc[max(0, i-20):i] if i > 0 else pd.DataFrame()
                
                features = {
                    'concurso': concurso.get('concurso', i),
                    
                    # Tendências de frequência
                    **self._calcular_tendencias_frequencia(numeros_atual, ultimos_5, ultimos_10, ultimos_20),
                    
                    # Números quentes e frios
                    **self._calcular_numeros_quentes_frios(numeros_atual, ultimos_10, ultimos_20),
                    
                    # Padrões sazonais
                    **self._calcular_padroes_sazonais(concurso),
                    
                    # Intervalos desde última aparição
                    **self._calcular_intervalos_aparicao(numeros_atual, dados_historicos.iloc[:i]),
                    
                    # Análise de repetições
                    **self._calcular_padroes_repeticao(numeros_atual, ultimos_5, ultimos_10)
                }
                
                features_temporais.append(features)
        
        df_features_temporais = pd.DataFrame(features_temporais)
        logger.info(f"Features temporais criadas: {len(df_features_temporais.columns)} colunas")
        
        return df_features_temporais
    
    def _calcular_tendencias_frequencia(self, numeros_atual: List[int], 
                                       ultimos_5: pd.DataFrame, 
                                       ultimos_10: pd.DataFrame,
                                       ultimos_20: pd.DataFrame) -> Dict:
        """
        Calcula tendências de frequência dos números
        """
        # Frequências em diferentes períodos
        freq_5 = self._calcular_frequencias_periodo(ultimos_5)
        freq_10 = self._calcular_frequencias_periodo(ultimos_10)
        freq_20 = self._calcular_frequencias_periodo(ultimos_20)
        
        features = {}
        
        # Frequência média dos números atuais em diferentes períodos
        if freq_5:
            features['freq_media_5'] = np.mean([freq_5.get(n, 0) for n in numeros_atual])
            features['freq_max_5'] = max([freq_5.get(n, 0) for n in numeros_atual])
            features['freq_min_5'] = min([freq_5.get(n, 0) for n in numeros_atual])
        else:
            features.update({'freq_media_5': 0, 'freq_max_5': 0, 'freq_min_5': 0})
        
        if freq_10:
            features['freq_media_10'] = np.mean([freq_10.get(n, 0) for n in numeros_atual])
            features['freq_max_10'] = max([freq_10.get(n, 0) for n in numeros_atual])
            features['freq_min_10'] = min([freq_10.get(n, 0) for n in numeros_atual])
        else:
            features.update({'freq_media_10': 0, 'freq_max_10': 0, 'freq_min_10': 0})
        
        if freq_20:
            features['freq_media_20'] = np.mean([freq_20.get(n, 0) for n in numeros_atual])
            features['freq_max_20'] = max([freq_20.get(n, 0) for n in numeros_atual])
            features['freq_min_20'] = min([freq_20.get(n, 0) for n in numeros_atual])
        else:
            features.update({'freq_media_20': 0, 'freq_max_20': 0, 'freq_min_20': 0})
        
        # Tendência (comparação entre períodos)
        if freq_10 and freq_20:
            tendencia = np.mean([freq_10.get(n, 0) - freq_20.get(n, 0) for n in numeros_atual])
            features['tendencia_freq'] = tendencia
        else:
            features['tendencia_freq'] = 0
        
        return features
    
    def _calcular_numeros_quentes_frios(self, numeros_atual: List[int],
                                       ultimos_10: pd.DataFrame,
                                       ultimos_20: pd.DataFrame) -> Dict:
        """
        Identifica números quentes (frequentes) e frios (raros)
        """
        freq_10 = self._calcular_frequencias_periodo(ultimos_10)
        freq_20 = self._calcular_frequencias_periodo(ultimos_20)
        
        features = {}
        
        if freq_10:
            # Definir limites para números quentes e frios
            freqs_valores = list(freq_10.values())
            if freqs_valores:
                limite_quente = np.percentile(freqs_valores, 75)
                limite_frio = np.percentile(freqs_valores, 25)
                
                # Contar números quentes e frios no jogo atual
                features['numeros_quentes_10'] = sum(1 for n in numeros_atual if freq_10.get(n, 0) >= limite_quente)
                features['numeros_frios_10'] = sum(1 for n in numeros_atual if freq_10.get(n, 0) <= limite_frio)
                features['numeros_medios_10'] = 15 - features['numeros_quentes_10'] - features['numeros_frios_10']
            else:
                features.update({'numeros_quentes_10': 0, 'numeros_frios_10': 0, 'numeros_medios_10': 15})
        else:
            features.update({'numeros_quentes_10': 0, 'numeros_frios_10': 0, 'numeros_medios_10': 15})
        
        if freq_20:
            freqs_valores = list(freq_20.values())
            if freqs_valores:
                limite_quente = np.percentile(freqs_valores, 75)
                limite_frio = np.percentile(freqs_valores, 25)
                
                features['numeros_quentes_20'] = sum(1 for n in numeros_atual if freq_20.get(n, 0) >= limite_quente)
                features['numeros_frios_20'] = sum(1 for n in numeros_atual if freq_20.get(n, 0) <= limite_frio)
                features['numeros_medios_20'] = 15 - features['numeros_quentes_20'] - features['numeros_frios_20']
            else:
                features.update({'numeros_quentes_20': 0, 'numeros_frios_20': 0, 'numeros_medios_20': 15})
        else:
            features.update({'numeros_quentes_20': 0, 'numeros_frios_20': 0, 'numeros_medios_20': 15})
        
        return features
    
    def _calcular_padroes_sazonais(self, concurso: pd.Series) -> Dict:
        """
        Calcula padrões sazonais baseados em data
        """
        features = {}
        
        # Tentar extrair data do concurso
        data_concurso = None
        for col in ['data', 'data_sorteio', 'date']:
            if col in concurso and pd.notna(concurso[col]):
                try:
                    if isinstance(concurso[col], str):
                        # Tentar diferentes formatos de data
                        for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
                            try:
                                data_concurso = datetime.strptime(concurso[col], fmt)
                                break
                            except:
                                continue
                    else:
                        data_concurso = pd.to_datetime(concurso[col])
                    break
                except:
                    continue
        
        if data_concurso:
            features.update({
                'dia_semana': data_concurso.weekday(),  # 0=segunda, 6=domingo
                'mes': data_concurso.month,
                'trimestre': (data_concurso.month - 1) // 3 + 1,
                'semestre': 1 if data_concurso.month <= 6 else 2,
                'dia_mes': data_concurso.day,
                'semana_ano': data_concurso.isocalendar()[1],
                'eh_inicio_mes': 1 if data_concurso.day <= 7 else 0,
                'eh_fim_mes': 1 if data_concurso.day >= 24 else 0
            })
        else:
            # Valores padrão se não houver data
            features.update({
                'dia_semana': 0, 'mes': 1, 'trimestre': 1, 'semestre': 1,
                'dia_mes': 1, 'semana_ano': 1, 'eh_inicio_mes': 0, 'eh_fim_mes': 0
            })
        
        return features
    
    def _calcular_intervalos_aparicao(self, numeros_atual: List[int], 
                                     historico_anterior: pd.DataFrame) -> Dict:
        """
        Calcula intervalos desde a última aparição de cada número
        """
        if historico_anterior.empty:
            return {
                'intervalo_medio': 0,
                'intervalo_max': 0,
                'intervalo_min': 0,
                'numeros_recentes': 0,  # apareceram nos últimos 3 concursos
                'numeros_antigos': 0    # não aparecem há mais de 10 concursos
            }
        
        intervalos = []
        
        # Otimização: criar dicionário de última aparição para evitar loop aninhado O(n²)
        ultima_aparicao = {}
        
        # Percorrer histórico uma única vez para mapear última aparição de cada número
        for i, concurso_hist in enumerate(historico_anterior.itertuples()):
            numeros_hist = self._extrair_numeros_concurso(concurso_hist)
            for numero in numeros_hist:
                ultima_aparicao[numero] = i
        
        # Calcular intervalos usando o dicionário (O(n) em vez de O(n²))
        for numero in numeros_atual:
            if numero in ultima_aparicao:
                intervalo = len(historico_anterior) - 1 - ultima_aparicao[numero]
            else:
                intervalo = len(historico_anterior)  # Nunca apareceu no histórico
            intervalos.append(intervalo)
        
        if intervalos:
            return {
                'intervalo_medio': np.mean(intervalos),
                'intervalo_max': max(intervalos),
                'intervalo_min': min(intervalos),
                'numeros_recentes': sum(1 for i in intervalos if i <= 3),
                'numeros_antigos': sum(1 for i in intervalos if i >= 10)
            }
        else:
            return {
                'intervalo_medio': 0, 'intervalo_max': 0, 'intervalo_min': 0,
                'numeros_recentes': 0, 'numeros_antigos': 0
            }
    
    def _calcular_padroes_repeticao(self, numeros_atual: List[int],
                                   ultimos_5: pd.DataFrame,
                                   ultimos_10: pd.DataFrame) -> Dict:
        """
        Calcula padrões de repetição com concursos anteriores
        """
        features = {
            'repetidos_ultimo_1': 0,
            'repetidos_ultimos_3': 0,
            'repetidos_ultimos_5': 0,
            'repetidos_ultimos_10': 0
        }
        
        # Repetições com último concurso
        if not ultimos_5.empty:
            ultimo_concurso = ultimos_5.iloc[-1]
            numeros_ultimo = self._extrair_numeros_concurso(ultimo_concurso)
            if numeros_ultimo:
                features['repetidos_ultimo_1'] = len(set(numeros_atual) & set(numeros_ultimo))
        
        # Repetições com últimos 3 concursos
        if len(ultimos_5) >= 3:
            numeros_ultimos_3 = set()
            for i in range(max(0, len(ultimos_5)-3), len(ultimos_5)):
                nums = self._extrair_numeros_concurso(ultimos_5.iloc[i])
                numeros_ultimos_3.update(nums)
            features['repetidos_ultimos_3'] = len(set(numeros_atual) & numeros_ultimos_3)
        
        # Repetições com últimos 5 concursos
        if not ultimos_5.empty:
            numeros_ultimos_5 = set()
            for _, concurso in ultimos_5.iterrows():
                nums = self._extrair_numeros_concurso(concurso)
                numeros_ultimos_5.update(nums)
            features['repetidos_ultimos_5'] = len(set(numeros_atual) & numeros_ultimos_5)
        
        # Repetições com últimos 10 concursos
        if not ultimos_10.empty:
            numeros_ultimos_10 = set()
            for _, concurso in ultimos_10.iterrows():
                nums = self._extrair_numeros_concurso(concurso)
                numeros_ultimos_10.update(nums)
            features['repetidos_ultimos_10'] = len(set(numeros_atual) & numeros_ultimos_10)
        
        return features
    
    def criar_features_padroes(self, dados):
        """Cria features baseadas em padrões dos números"""
        features_padroes = {}
        
        if len(dados) == 0:
            return features_padroes
        
        # Análise do último jogo
        ultimo_jogo = dados.iloc[-1] if hasattr(dados, 'iloc') else dados[-1]
        numeros = self._extrair_numeros_concurso(ultimo_jogo)
        
        # Features de consecutivos
        features_padroes.update(self._calcular_consecutivos_avancados(numeros))
        
        # Features de gaps
        features_padroes.update(self._calcular_gaps_avancados(numeros))
        
        # Features de ciclos (análise histórica)
        if len(dados) > 10:
            features_padroes.update(self._calcular_ciclos(dados))
        
        # Features de simetria
        features_padroes.update(self._calcular_simetria(numeros))
        
        return features_padroes
    
    def extrair_numeros_concurso(self, linha_dados) -> List[int]:
        """Extrai os números de um concurso da linha de dados"""
        try:
            # Os números estão nas colunas B1-B15
            colunas_numeros = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                             'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15']
            
            if hasattr(linha_dados, 'name'):  # É uma Series do pandas
                numeros = [linha_dados[col] for col in colunas_numeros if col in linha_dados.index]
            elif isinstance(linha_dados, dict):
                numeros = [linha_dados[col] for col in colunas_numeros if col in linha_dados]
            else:
                # Assumir que é um DataFrame e pegar a primeira linha
                if hasattr(linha_dados, 'iloc'):
                    linha = linha_dados.iloc[0] if len(linha_dados) > 0 else linha_dados
                    numeros = [linha[col] for col in colunas_numeros if col in linha.index]
                else:
                    return []
            
            # Converter para lista de inteiros, removendo NaN
            numeros_limpos = []
            for x in numeros:
                if pd.notna(x) and x > 0:
                    try:
                        num = int(x)
                        if 1 <= num <= 25:
                            numeros_limpos.append(num)
                    except (ValueError, TypeError):
                        continue
            
            # Validar se temos exatamente 15 números
            if len(numeros_limpos) != 15:
                return []
            
            return sorted(numeros_limpos)
            
        except Exception as e:
            return []
    
    def _calcular_consecutivos_avancados(self, numeros):
        """Calcula features relacionadas a números consecutivos"""
        numeros_sorted = sorted(numeros)
        
        # Contar sequências consecutivas
        consecutivos = 0
        max_consecutivos = 0
        atual_consecutivos = 1
        
        for i in range(1, len(numeros_sorted)):
            if numeros_sorted[i] == numeros_sorted[i-1] + 1:
                atual_consecutivos += 1
                consecutivos += 1
            else:
                max_consecutivos = max(max_consecutivos, atual_consecutivos)
                atual_consecutivos = 1
        
        max_consecutivos = max(max_consecutivos, atual_consecutivos)
        
        return {
            'total_consecutivos': consecutivos,
            'max_sequencia_consecutiva': max_consecutivos,
            'tem_consecutivos': 1 if consecutivos > 0 else 0,
            'densidade_consecutivos': consecutivos / len(numeros) if len(numeros) > 0 else 0
        }
    
    def _calcular_gaps_avancados(self, numeros):
        """Calcula análise avançada de gaps entre números"""
        numeros_sorted = sorted(numeros)
        gaps = [numeros_sorted[i+1] - numeros_sorted[i] for i in range(len(numeros_sorted)-1)]
        
        if not gaps:
            return {}
        
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)
        
        return {
            'gap_medio': gap_mean,
            'gap_mediano': np.median(gaps),
            'gap_std': gap_std,
            'gap_min': min(gaps),
            'gap_max': max(gaps),
            'gap_range': max(gaps) - min(gaps),
            'gaps_pequenos': sum(1 for g in gaps if g <= 2),  # gaps <= 2
            'gaps_grandes': sum(1 for g in gaps if g >= 5),   # gaps >= 5
            'uniformidade_gaps': 1 - (gap_std / gap_mean) if gap_mean > 0 else 0
        }
    
    def _calcular_ciclos(self, dados):
        """Analisa padrões cíclicos nos dados históricos"""
        features_ciclos = {}
        
        # Extrair todos os números dos últimos jogos
        ultimos_jogos = dados.tail(20) if hasattr(dados, 'tail') else dados[-20:]
        
        # Frequência de aparição nos últimos jogos
        freq_recente = {}
        for _, jogo in ultimos_jogos.iterrows():
            numeros = self._extrair_numeros_concurso(jogo)
            for num in numeros:
                freq_recente[num] = freq_recente.get(num, 0) + 1
        
        # Análise de ciclos
        numeros_frequentes = [num for num, freq in freq_recente.items() if freq >= 3]
        numeros_raros = [num for num, freq in freq_recente.items() if freq <= 1]
        
        features_ciclos.update({
            'numeros_em_ciclo_alto': len(numeros_frequentes),
            'numeros_em_ciclo_baixo': len(numeros_raros),
            'razao_ciclo_alto_baixo': len(numeros_frequentes) / max(len(numeros_raros), 1),
            'diversidade_ciclos': len(set(freq_recente.values()))
        })
        
        return features_ciclos
    
    def _calcular_simetria(self, numeros):
        """Calcula features de simetria e distribuição espacial"""
        if not numeros or len(numeros) == 0:
            return {
                'simetria_metades': 0,
                'razao_primeira_metade': 0,
                'razao_segunda_metade': 0,
                'distribuicao_tercos_1': 0,
                'distribuicao_tercos_2': 0,
                'distribuicao_tercos_3': 0,
                'equilibrio_tercos': 0,
                'concentracao_centro': 0
            }
        
        # Dividir em metades
        primeira_metade = [n for n in numeros if n <= 12]
        segunda_metade = [n for n in numeros if n > 12]
        
        # Análise por terços
        primeiro_terco = [n for n in numeros if n <= 8]
        segundo_terco = [n for n in numeros if 9 <= n <= 16]
        terceiro_terco = [n for n in numeros if n >= 17]
        
        total_numeros = len(numeros)
        
        return {
            'simetria_metades': abs(len(primeira_metade) - len(segunda_metade)),
            'razao_primeira_metade': len(primeira_metade) / total_numeros,
            'razao_segunda_metade': len(segunda_metade) / total_numeros,
            'distribuicao_tercos_1': len(primeiro_terco),
            'distribuicao_tercos_2': len(segundo_terco),
            'distribuicao_tercos_3': len(terceiro_terco),
            'equilibrio_tercos': np.std([len(primeiro_terco), len(segundo_terco), len(terceiro_terco)]),
            'concentracao_centro': len(segundo_terco) / total_numeros
        }

    def criar_features_completas(self, dados):
        """Cria todas as features disponíveis"""
        features = {}
        
        # Features estatísticas
        features_estat = self.criar_features_estatisticas(dados)
        features.update(features_estat)
        
        # Features temporais (se dados históricos disponíveis)
        if len(dados) > 1:
            features_temp = self.criar_features_temporais(dados)
            features.update(features_temp)
        
        # Features de padrões
        features_padr = self.criar_features_padroes(dados)
        features.update(features_padr)
        
        return features
    
    def _calcular_frequencias_periodo(self, dados_periodo: pd.DataFrame) -> Dict[int, int]:
        """
        Calcula frequências dos números em um período específico - OTIMIZADO
        """
        # Cache para evitar recálculos desnecessários
        cache_key = f"{len(dados_periodo)}_{dados_periodo.index[0] if not dados_periodo.empty else 0}_{dados_periodo.index[-1] if not dados_periodo.empty else 0}"
        
        if hasattr(self, '_freq_cache') and cache_key in self._freq_cache:
            return self._freq_cache[cache_key]
        
        if not hasattr(self, '_freq_cache'):
            self._freq_cache = {}
        
        frequencias = {i: 0 for i in range(1, 26)}
        
        # Otimização: processar em lotes se o dataset for muito grande
        if len(dados_periodo) > 100:
            # Para períodos grandes, usar vectorização
            for _, concurso in dados_periodo.iterrows():
                numeros = self._extrair_numeros_concurso(concurso)
                for numero in numeros:
                    if 1 <= numero <= 25:
                        frequencias[numero] += 1
        else:
            # Para períodos pequenos, manter abordagem original
            for _, concurso in dados_periodo.iterrows():
                numeros = self._extrair_numeros_concurso(concurso)
                for numero in numeros:
                    if 1 <= numero <= 25:
                        frequencias[numero] += 1
        
        # Armazenar no cache
        self._freq_cache[cache_key] = frequencias
        
        # Limitar tamanho do cache
        if len(self._freq_cache) > 1000:
            # Remover entradas mais antigas
            oldest_keys = list(self._freq_cache.keys())[:500]
            for key in oldest_keys:
                del self._freq_cache[key]
        
        return frequencias
    
    def normalizar_features(self, features_df: pd.DataFrame, metodo: str = 'standard') -> pd.DataFrame:
        """Normaliza as features usando diferentes métodos"""
        if metodo not in self.scalers:
            logger.warning(f"Método {metodo} não disponível. Usando 'standard'.")
            metodo = 'standard'
        
        scaler = self.scalers[metodo]
        features_normalizadas = scaler.fit_transform(features_df)
        
        return pd.DataFrame(features_normalizadas, columns=features_df.columns, index=features_df.index)
    
    def selecionar_features(self, features_df: pd.DataFrame, target: pd.Series, k: int = 50) -> pd.DataFrame:
        """Seleciona as k melhores features usando diferentes critérios"""
        logger.info(f"Selecionando {k} melhores features...")
        
        # Usar SelectKBest com f_regression
        selector = SelectKBest(score_func=f_regression, k=min(k, features_df.shape[1]))
        features_selecionadas = selector.fit_transform(features_df, target)
        
        # Obter nomes das features selecionadas
        feature_mask = selector.get_support()
        feature_names = features_df.columns[feature_mask].tolist()
        
        self.feature_names = feature_names
        self.feature_selector = selector
        
        logger.info(f"Features selecionadas: {len(feature_names)}")
        
        return pd.DataFrame(features_selecionadas, columns=feature_names, index=features_df.index)
    
    def obter_importancia_features(self, features_df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Calcula importância das features usando Random Forest"""
        logger.info("Calculando importância das features...")
        
        # Treinar Random Forest para obter importâncias
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(features_df, target)
        
        # Criar DataFrame com importâncias
        importancias = pd.DataFrame({
            'feature': features_df.columns,
            'importancia': rf.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        return importancias
    
    def pipeline_completo(self, dados_historicos: pd.DataFrame, 
                         normalizar: bool = True,
                         selecionar: bool = True,
                         k_features: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pipeline completo de feature engineering"""
        logger.info("Iniciando pipeline completo de feature engineering...")
        
        # 1. Criar features estatísticas
        features_estat = self.criar_features_estatisticas(dados_historicos)
        
        # 2. Criar features temporais
        features_temp = self.criar_features_temporais(dados_historicos)
        
        # 3. Combinar todas as features
        features_completas = pd.merge(features_estat, features_temp, on='concurso', how='inner')
        
        # 4. Criar target (próximo concurso)
        # Para este exemplo, vamos usar a soma dos números como target
        target = features_completas['soma_total'].shift(-1).dropna()
        features_completas = features_completas.iloc[:-1]  # Remover última linha sem target
        
        # 5. Remover colunas não numéricas
        numeric_columns = features_completas.select_dtypes(include=[np.number]).columns
        features_numericas = features_completas[numeric_columns]
        
        # 6. Tratar valores NaN
        features_numericas = features_numericas.fillna(features_numericas.median())
        
        # 7. Normalizar se solicitado
        if normalizar:
            features_numericas = self.normalizar_features(features_numericas)
        
        # 8. Selecionar features se solicitado
        if selecionar and len(features_numericas.columns) > k_features:
            features_numericas = self.selecionar_features(features_numericas, target, k_features)
        
        # 9. Obter importância das features
        importancias = self.obter_importancia_features(features_numericas, target)
        
        logger.info(f"Pipeline concluído. Features finais: {features_numericas.shape[1]}")
        
        return features_numericas, importancias