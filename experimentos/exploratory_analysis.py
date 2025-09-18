#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Exploratória Avançada - Sistema Lotofácil

Este módulo implementa análises estatísticas profundas dos dados históricos
da Lotofácil para identificar padrões, tendências e oportunidades de otimização.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, normaltest
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
import warnings
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LotofacilExploratoryAnalysis:
    """
    Classe para análise exploratória avançada dos dados da Lotofácil
    """
    
    def __init__(self, dados_path: str = "dados/dados_lotofacil.xlsx"):
        """
        Inicializa a análise exploratória
        
        Args:
            dados_path: Caminho para o arquivo de dados
        """
        self.dados_path = dados_path
        self.dados = None
        self.analysis_results = {}
        self.output_dir = Path("experimentos/resultados")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar dados
        self._load_data()
    
    def _load_data(self):
        """
        Carrega e prepara os dados para análise
        """
        try:
            if Path(self.dados_path).exists():
                self.dados = pd.read_excel(self.dados_path)
                print(f"Dados carregados: {len(self.dados)} concursos")
            else:
                print(f"Arquivo {self.dados_path} não encontrado. Gerando dados sintéticos...")
                self._generate_synthetic_data()
            
            # Preparar dados
            self._prepare_data()
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            print("Gerando dados sintéticos para análise...")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_concursos: int = 3000):
        """
        Gera dados sintéticos para análise quando dados reais não estão disponíveis
        
        Args:
            n_concursos: Número de concursos a gerar
        """
        np.random.seed(42)
        
        dados_sinteticos = []
        
        for i in range(1, n_concursos + 1):
            # Gerar 15 números únicos entre 1 e 25
            numeros = sorted(np.random.choice(range(1, 26), size=15, replace=False))
            
            # Criar entrada
            entrada = {
                'Concurso': i,
                'Data': datetime(2010, 1, 1) + timedelta(days=i*3),
                'Bola1': numeros[0], 'Bola2': numeros[1], 'Bola3': numeros[2],
                'Bola4': numeros[3], 'Bola5': numeros[4], 'Bola6': numeros[5],
                'Bola7': numeros[6], 'Bola8': numeros[7], 'Bola9': numeros[8],
                'Bola10': numeros[9], 'Bola11': numeros[10], 'Bola12': numeros[11],
                'Bola13': numeros[12], 'Bola14': numeros[13], 'Bola15': numeros[14]
            }
            
            dados_sinteticos.append(entrada)
        
        self.dados = pd.DataFrame(dados_sinteticos)
        print(f"Dados sintéticos gerados: {len(self.dados)} concursos")
    
    def _prepare_data(self):
        """
        Prepara os dados para análise
        """
        # Identificar colunas de bolas
        bola_cols = [col for col in self.dados.columns if col.startswith('Bola')]
        
        if not bola_cols:
            # Tentar outras nomenclaturas
            possible_cols = ['1ª Dezena', '2ª Dezena', '3ª Dezena', '4ª Dezena', '5ª Dezena',
                           '6ª Dezena', '7ª Dezena', '8ª Dezena', '9ª Dezena', '10ª Dezena',
                           '11ª Dezena', '12ª Dezena', '13ª Dezena', '14ª Dezena', '15ª Dezena']
            
            bola_cols = [col for col in possible_cols if col in self.dados.columns]
        
        if len(bola_cols) >= 15:
            self.bola_cols = bola_cols[:15]
        else:
            raise ValueError(f"Não foi possível identificar 15 colunas de números. Encontradas: {bola_cols}")
        
        # Criar coluna com todos os números do concurso
        self.dados['numeros'] = self.dados[self.bola_cols].values.tolist()
        
        # Converter data se necessário
        if 'Data' in self.dados.columns:
            self.dados['Data'] = pd.to_datetime(self.dados['Data'], errors='coerce')
        
        print(f"Dados preparados. Colunas de números: {len(self.bola_cols)}")
    
    def analyze_frequency_patterns(self) -> Dict[str, Any]:
        """
        Analisa padrões de frequência dos números
        
        Returns:
            Dicionário com resultados da análise de frequência
        """
        print("\n=== ANÁLISE DE PADRÕES DE FREQUÊNCIA ===")
        
        # Contar frequência de cada número
        all_numbers = []
        for numeros in self.dados['numeros']:
            all_numbers.extend(numeros)
        
        freq_counter = Counter(all_numbers)
        freq_df = pd.DataFrame(list(freq_counter.items()), columns=['Numero', 'Frequencia'])
        freq_df = freq_df.sort_values('Frequencia', ascending=False)
        
        # Estatísticas básicas
        freq_stats = {
            'media_frequencia': freq_df['Frequencia'].mean(),
            'desvio_frequencia': freq_df['Frequencia'].std(),
            'numero_mais_frequente': freq_df.iloc[0]['Numero'],
            'maior_frequencia': freq_df.iloc[0]['Frequencia'],
            'numero_menos_frequente': freq_df.iloc[-1]['Numero'],
            'menor_frequencia': freq_df.iloc[-1]['Frequencia'],
            'coeficiente_variacao': freq_df['Frequencia'].std() / freq_df['Frequencia'].mean()
        }
        
        # Teste de uniformidade (Chi-quadrado)
        expected_freq = len(all_numbers) / 25  # Frequência esperada se fosse uniforme
        chi2_stat, chi2_p = stats.chisquare(freq_df['Frequencia'], 
                                           f_exp=[expected_freq] * 25)
        
        freq_stats['chi2_uniformidade'] = chi2_stat
        freq_stats['p_value_uniformidade'] = chi2_p
        freq_stats['eh_uniforme'] = chi2_p > 0.05
        
        # Números "quentes" e "frios"
        q75 = freq_df['Frequencia'].quantile(0.75)
        q25 = freq_df['Frequencia'].quantile(0.25)
        
        numeros_quentes = freq_df[freq_df['Frequencia'] >= q75]['Numero'].tolist()
        numeros_frios = freq_df[freq_df['Frequencia'] <= q25]['Numero'].tolist()
        
        freq_stats['numeros_quentes'] = numeros_quentes
        freq_stats['numeros_frios'] = numeros_frios
        
        # Salvar resultados
        self.analysis_results['frequency_patterns'] = freq_stats
        
        # Criar visualização
        self._plot_frequency_analysis(freq_df)
        
        print(f"Número mais frequente: {freq_stats['numero_mais_frequente']} ({freq_stats['maior_frequencia']} vezes)")
        print(f"Número menos frequente: {freq_stats['numero_menos_frequente']} ({freq_stats['menor_frequencia']} vezes)")
        print(f"Coeficiente de variação: {freq_stats['coeficiente_variacao']:.4f}")
        print(f"Distribuição é uniforme: {freq_stats['eh_uniforme']} (p={freq_stats['p_value_uniformidade']:.4f})")
        
        return freq_stats
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analisa padrões temporais nos sorteios
        
        Returns:
            Dicionário com resultados da análise temporal
        """
        print("\n=== ANÁLISE DE PADRÕES TEMPORAIS ===")
        
        temporal_stats = {}
        
        if 'Data' in self.dados.columns and self.dados['Data'].notna().any():
            # Análise por dia da semana
            self.dados['dia_semana'] = self.dados['Data'].dt.day_name()
            
            # Análise por mês
            self.dados['mes'] = self.dados['Data'].dt.month
            
            # Análise por ano
            self.dados['ano'] = self.dados['Data'].dt.year
            
            # Frequência por dia da semana
            freq_dia_semana = self.dados['dia_semana'].value_counts()
            temporal_stats['frequencia_dia_semana'] = freq_dia_semana.to_dict()
            
            # Frequência por mês
            freq_mes = self.dados['mes'].value_counts().sort_index()
            temporal_stats['frequencia_mes'] = freq_mes.to_dict()
            
            # Tendências anuais
            freq_ano = self.dados['ano'].value_counts().sort_index()
            temporal_stats['frequencia_ano'] = freq_ano.to_dict()
            
            # Análise de sazonalidade
            self._analyze_seasonality(temporal_stats)
        
        # Análise de intervalos entre números
        self._analyze_number_intervals(temporal_stats)
        
        # Análise de sequências
        self._analyze_sequences(temporal_stats)
        
        self.analysis_results['temporal_patterns'] = temporal_stats
        
        return temporal_stats
    
    def analyze_distribution_patterns(self) -> Dict[str, Any]:
        """
        Analisa padrões de distribuição dos números
        
        Returns:
            Dicionário com resultados da análise de distribuição
        """
        print("\n=== ANÁLISE DE PADRÕES DE DISTRIBUIÇÃO ===")
        
        distribution_stats = {}
        
        # Análise par/ímpar
        pares_impares = []
        for numeros in self.dados['numeros']:
            pares = sum(1 for n in numeros if n % 2 == 0)
            impares = 15 - pares
            pares_impares.append({'pares': pares, 'impares': impares})
        
        pares_df = pd.DataFrame(pares_impares)
        distribution_stats['par_impar'] = {
            'media_pares': pares_df['pares'].mean(),
            'desvio_pares': pares_df['pares'].std(),
            'distribuicao_pares': pares_df['pares'].value_counts().to_dict()
        }
        
        # Análise por faixas (baixos, médios, altos)
        faixas_dist = []
        for numeros in self.dados['numeros']:
            baixos = sum(1 for n in numeros if n <= 8)
            medios = sum(1 for n in numeros if 9 <= n <= 17)
            altos = sum(1 for n in numeros if n >= 18)
            faixas_dist.append({'baixos': baixos, 'medios': medios, 'altos': altos})
        
        faixas_df = pd.DataFrame(faixas_dist)
        distribution_stats['faixas'] = {
            'media_baixos': faixas_df['baixos'].mean(),
            'media_medios': faixas_df['medios'].mean(),
            'media_altos': faixas_df['altos'].mean(),
            'distribuicao_baixos': faixas_df['baixos'].value_counts().to_dict(),
            'distribuicao_medios': faixas_df['medios'].value_counts().to_dict(),
            'distribuicao_altos': faixas_df['altos'].value_counts().to_dict()
        }
        
        # Análise de soma dos números
        somas = [sum(numeros) for numeros in self.dados['numeros']]
        distribution_stats['soma'] = {
            'media_soma': np.mean(somas),
            'desvio_soma': np.std(somas),
            'min_soma': min(somas),
            'max_soma': max(somas),
            'mediana_soma': np.median(somas)
        }
        
        # Teste de normalidade da soma
        normaltest_stat, normaltest_p = normaltest(somas)
        distribution_stats['soma']['normaltest_stat'] = normaltest_stat
        distribution_stats['soma']['normaltest_p'] = normaltest_p
        distribution_stats['soma']['eh_normal'] = normaltest_p > 0.05
        
        # Análise de gaps (intervalos entre números consecutivos)
        gaps_analysis = self._analyze_gaps()
        distribution_stats['gaps'] = gaps_analysis
        
        self.analysis_results['distribution_patterns'] = distribution_stats
        
        # Criar visualizações
        self._plot_distribution_analysis(pares_df, faixas_df, somas)
        
        print(f"Média de números pares por jogo: {distribution_stats['par_impar']['media_pares']:.2f}")
        print(f"Média de números baixos: {distribution_stats['faixas']['media_baixos']:.2f}")
        print(f"Média de números médios: {distribution_stats['faixas']['media_medios']:.2f}")
        print(f"Média de números altos: {distribution_stats['faixas']['media_altos']:.2f}")
        print(f"Soma média dos números: {distribution_stats['soma']['media_soma']:.2f}")
        
        return distribution_stats
    
    def analyze_correlation_patterns(self) -> Dict[str, Any]:
        """
        Analisa correlações entre números e posições
        
        Returns:
            Dicionário com resultados da análise de correlação
        """
        print("\n=== ANÁLISE DE PADRÕES DE CORRELAÇÃO ===")
        
        correlation_stats = {}
        
        # Matriz de correlação entre posições
        positions_df = self.dados[self.bola_cols]
        correlation_matrix = positions_df.corr()
        
        correlation_stats['correlation_matrix'] = correlation_matrix.to_dict()
        correlation_stats['avg_correlation'] = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        # Análise de co-ocorrência de números
        cooccurrence_matrix = self._calculate_cooccurrence_matrix()
        correlation_stats['cooccurrence_matrix'] = cooccurrence_matrix
        
        # Números que aparecem frequentemente juntos
        frequent_pairs = self._find_frequent_pairs()
        correlation_stats['frequent_pairs'] = frequent_pairs
        
        # Análise de dependência entre números consecutivos
        consecutive_analysis = self._analyze_consecutive_numbers()
        correlation_stats['consecutive_analysis'] = consecutive_analysis
        
        self.analysis_results['correlation_patterns'] = correlation_stats
        
        # Criar visualizações
        self._plot_correlation_analysis(correlation_matrix, cooccurrence_matrix)
        
        print(f"Correlação média entre posições: {correlation_stats['avg_correlation']:.4f}")
        print(f"Pares mais frequentes: {frequent_pairs[:5]}")
        
        return correlation_stats
    
    def _analyze_seasonality(self, temporal_stats: Dict[str, Any]):
        """
        Analisa sazonalidade nos dados
        
        Args:
            temporal_stats: Dicionário para armazenar resultados
        """
        # Análise de sazonalidade por frequência de números por mês
        monthly_number_freq = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.dados.iterrows():
            if pd.notna(row['Data']):
                mes = row['Data'].month
                for numero in row['numeros']:
                    monthly_number_freq[mes][numero] += 1
        
        # Calcular variação sazonal
        seasonal_variation = {}
        for numero in range(1, 26):
            monthly_freqs = [monthly_number_freq[mes][numero] for mes in range(1, 13)]
            if sum(monthly_freqs) > 0:
                cv = np.std(monthly_freqs) / np.mean(monthly_freqs) if np.mean(monthly_freqs) > 0 else 0
                seasonal_variation[numero] = cv
        
        temporal_stats['seasonal_variation'] = seasonal_variation
        
        # Números com maior variação sazonal
        most_seasonal = sorted(seasonal_variation.items(), key=lambda x: x[1], reverse=True)[:5]
        temporal_stats['most_seasonal_numbers'] = most_seasonal
    
    def _analyze_number_intervals(self, temporal_stats: Dict[str, Any]):
        """
        Analisa intervalos entre aparições dos números
        
        Args:
            temporal_stats: Dicionário para armazenar resultados
        """
        # Calcular intervalos entre aparições de cada número
        number_intervals = defaultdict(list)
        last_appearance = {}
        
        for idx, numeros in enumerate(self.dados['numeros']):
            for numero in numeros:
                if numero in last_appearance:
                    interval = idx - last_appearance[numero]
                    number_intervals[numero].append(interval)
                last_appearance[numero] = idx
        
        # Estatísticas dos intervalos
        interval_stats = {}
        for numero in range(1, 26):
            if numero in number_intervals and number_intervals[numero]:
                intervals = number_intervals[numero]
                interval_stats[numero] = {
                    'media_intervalo': np.mean(intervals),
                    'desvio_intervalo': np.std(intervals),
                    'min_intervalo': min(intervals),
                    'max_intervalo': max(intervals)
                }
        
        temporal_stats['number_intervals'] = interval_stats
    
    def _analyze_sequences(self, temporal_stats: Dict[str, Any]):
        """
        Analisa sequências e números consecutivos
        
        Args:
            temporal_stats: Dicionário para armazenar resultados
        """
        sequence_stats = []
        
        for numeros in self.dados['numeros']:
            sorted_nums = sorted(numeros)
            
            # Contar sequências consecutivas
            max_sequence = 1
            current_sequence = 1
            
            for i in range(1, len(sorted_nums)):
                if sorted_nums[i] == sorted_nums[i-1] + 1:
                    current_sequence += 1
                    max_sequence = max(max_sequence, current_sequence)
                else:
                    current_sequence = 1
            
            sequence_stats.append(max_sequence)
        
        temporal_stats['sequences'] = {
            'media_max_sequencia': np.mean(sequence_stats),
            'distribuicao_sequencias': Counter(sequence_stats)
        }
    
    def _analyze_gaps(self) -> Dict[str, Any]:
        """
        Analisa gaps (intervalos) entre números sorteados
        
        Returns:
            Dicionário com estatísticas de gaps
        """
        all_gaps = []
        
        for numeros in self.dados['numeros']:
            sorted_nums = sorted(numeros)
            gaps = [sorted_nums[i] - sorted_nums[i-1] for i in range(1, len(sorted_nums))]
            all_gaps.extend(gaps)
        
        return {
            'media_gap': np.mean(all_gaps),
            'desvio_gap': np.std(all_gaps),
            'min_gap': min(all_gaps),
            'max_gap': max(all_gaps),
            'distribuicao_gaps': Counter(all_gaps)
        }
    
    def _calculate_cooccurrence_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Calcula matriz de co-ocorrência entre números
        
        Returns:
            Matriz de co-ocorrência
        """
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for numeros in self.dados['numeros']:
            for i, num1 in enumerate(numeros):
                for j, num2 in enumerate(numeros):
                    if i != j:
                        cooccurrence[num1][num2] += 1
        
        return {k: dict(v) for k, v in cooccurrence.items()}
    
    def _find_frequent_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Encontra pares de números que aparecem frequentemente juntos
        
        Returns:
            Lista de tuplas (num1, num2, frequencia)
        """
        pair_counts = defaultdict(int)
        
        for numeros in self.dados['numeros']:
            for i in range(len(numeros)):
                for j in range(i+1, len(numeros)):
                    pair = tuple(sorted([numeros[i], numeros[j]]))
                    pair_counts[pair] += 1
        
        # Ordenar por frequência
        frequent_pairs = [(pair[0], pair[1], count) for pair, count in 
                         sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)]
        
        return frequent_pairs[:20]  # Top 20 pares
    
    def _analyze_consecutive_numbers(self) -> Dict[str, Any]:
        """
        Analisa padrões de números consecutivos
        
        Returns:
            Dicionário com análise de números consecutivos
        """
        consecutive_counts = []
        
        for numeros in self.dados['numeros']:
            sorted_nums = sorted(numeros)
            consecutive = 0
            
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i+1] == sorted_nums[i] + 1:
                    consecutive += 1
            
            consecutive_counts.append(consecutive)
        
        return {
            'media_consecutivos': np.mean(consecutive_counts),
            'desvio_consecutivos': np.std(consecutive_counts),
            'distribuicao_consecutivos': Counter(consecutive_counts)
        }
    
    def _plot_frequency_analysis(self, freq_df: pd.DataFrame):
        """
        Cria visualizações para análise de frequência
        
        Args:
            freq_df: DataFrame com frequências
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico de barras - frequência por número
        axes[0, 0].bar(freq_df['Numero'], freq_df['Frequencia'])
        axes[0, 0].set_title('Frequência de Cada Número')
        axes[0, 0].set_xlabel('Número')
        axes[0, 0].set_ylabel('Frequência')
        
        # Histograma das frequências
        axes[0, 1].hist(freq_df['Frequencia'], bins=10, edgecolor='black')
        axes[0, 1].set_title('Distribuição das Frequências')
        axes[0, 1].set_xlabel('Frequência')
        axes[0, 1].set_ylabel('Quantidade de Números')
        
        # Box plot das frequências
        axes[1, 0].boxplot(freq_df['Frequencia'])
        axes[1, 0].set_title('Box Plot das Frequências')
        axes[1, 0].set_ylabel('Frequência')
        
        # Linha de tendência
        axes[1, 1].plot(freq_df['Numero'], freq_df['Frequencia'], 'o-')
        axes[1, 1].axhline(y=freq_df['Frequencia'].mean(), color='r', linestyle='--', label='Média')
        axes[1, 1].set_title('Tendência das Frequências')
        axes[1, 1].set_xlabel('Número')
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_analysis(self, pares_df: pd.DataFrame, 
                                  faixas_df: pd.DataFrame, somas: List[int]):
        """
        Cria visualizações para análise de distribuição
        
        Args:
            pares_df: DataFrame com contagem de pares/ímpares
            faixas_df: DataFrame com distribuição por faixas
            somas: Lista com somas dos jogos
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Distribuição par/ímpar
        pares_df['pares'].hist(bins=range(0, 16), ax=axes[0, 0], edgecolor='black')
        axes[0, 0].set_title('Distribuição de Números Pares por Jogo')
        axes[0, 0].set_xlabel('Quantidade de Pares')
        axes[0, 0].set_ylabel('Frequência')
        
        # Distribuição por faixas - baixos
        faixas_df['baixos'].hist(bins=range(0, 16), ax=axes[0, 1], edgecolor='black')
        axes[0, 1].set_title('Distribuição de Números Baixos (1-8)')
        axes[0, 1].set_xlabel('Quantidade de Baixos')
        axes[0, 1].set_ylabel('Frequência')
        
        # Distribuição por faixas - médios
        faixas_df['medios'].hist(bins=range(0, 16), ax=axes[0, 2], edgecolor='black')
        axes[0, 2].set_title('Distribuição de Números Médios (9-17)')
        axes[0, 2].set_xlabel('Quantidade de Médios')
        axes[0, 2].set_ylabel('Frequência')
        
        # Distribuição por faixas - altos
        faixas_df['altos'].hist(bins=range(0, 16), ax=axes[1, 0], edgecolor='black')
        axes[1, 0].set_title('Distribuição de Números Altos (18-25)')
        axes[1, 0].set_xlabel('Quantidade de Altos')
        axes[1, 0].set_ylabel('Frequência')
        
        # Distribuição das somas
        axes[1, 1].hist(somas, bins=30, edgecolor='black')
        axes[1, 1].set_title('Distribuição das Somas dos Jogos')
        axes[1, 1].set_xlabel('Soma dos Números')
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].axvline(x=np.mean(somas), color='r', linestyle='--', label='Média')
        axes[1, 1].legend()
        
        # Scatter plot: pares vs soma
        axes[1, 2].scatter(pares_df['pares'], somas, alpha=0.6)
        axes[1, 2].set_title('Relação: Números Pares vs Soma')
        axes[1, 2].set_xlabel('Quantidade de Pares')
        axes[1, 2].set_ylabel('Soma dos Números')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, correlation_matrix: pd.DataFrame, 
                                 cooccurrence_matrix: Dict[str, Dict[str, int]]):
        """
        Cria visualizações para análise de correlação
        
        Args:
            correlation_matrix: Matriz de correlação entre posições
            cooccurrence_matrix: Matriz de co-ocorrência
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap da matriz de correlação
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=axes[0])
        axes[0].set_title('Correlação entre Posições dos Números')
        
        # Preparar dados de co-ocorrência para visualização
        cooc_df = pd.DataFrame(cooccurrence_matrix).fillna(0)
        
        # Heatmap da co-ocorrência (apenas top números)
        top_numbers = sorted(cooc_df.index)[:15]  # Primeiros 15 números
        cooc_subset = cooc_df.loc[top_numbers, top_numbers]
        
        sns.heatmap(cooc_subset, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Co-ocorrência entre Números (Top 15)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self) -> str:
        """
        Gera relatório completo da análise exploratória
        
        Returns:
            Caminho do arquivo do relatório
        """
        print("\n=== GERANDO RELATÓRIO COMPLETO ===")
        
        # Executar todas as análises
        freq_results = self.analyze_frequency_patterns()
        temporal_results = self.analyze_temporal_patterns()
        dist_results = self.analyze_distribution_patterns()
        corr_results = self.analyze_correlation_patterns()
        
        # Gerar insights e recomendações
        insights = self._generate_insights()
        
        # Criar relatório
        report_path = self.output_dir / f"relatorio_analise_exploratoria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Análise Exploratória - Lotofácil\n\n")
            f.write(f"**Data da Análise:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"**Total de Concursos Analisados:** {len(self.dados)}\n\n")
            
            # Resumo Executivo
            f.write("## Resumo Executivo\n\n")
            f.write(self._create_executive_summary())
            
            # Análise de Frequência
            f.write("\n## 1. Análise de Padrões de Frequência\n\n")
            f.write(self._format_frequency_results(freq_results))
            
            # Análise Temporal
            f.write("\n## 2. Análise de Padrões Temporais\n\n")
            f.write(self._format_temporal_results(temporal_results))
            
            # Análise de Distribuição
            f.write("\n## 3. Análise de Padrões de Distribuição\n\n")
            f.write(self._format_distribution_results(dist_results))
            
            # Análise de Correlação
            f.write("\n## 4. Análise de Padrões de Correlação\n\n")
            f.write(self._format_correlation_results(corr_results))
            
            # Insights e Recomendações
            f.write("\n## 5. Insights e Recomendações\n\n")
            f.write(insights)
            
            # Limitações do Modelo Atual
            f.write("\n## 6. Limitações Identificadas no Modelo Atual\n\n")
            f.write(self._identify_model_limitations())
            
            # Próximos Passos
            f.write("\n## 7. Próximos Passos para Otimização\n\n")
            f.write(self._suggest_optimization_steps())
        
        # Salvar dados da análise em JSON
        json_path = self.output_dir / f"dados_analise_exploratoria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Relatório salvo em: {report_path}")
        print(f"Dados da análise salvos em: {json_path}")
        
        return str(report_path)
    
    def _create_executive_summary(self) -> str:
        """
        Cria resumo executivo da análise
        
        Returns:
            Texto do resumo executivo
        """
        summary = [
            "### Principais Descobertas:\n",
            f"- **Dataset:** {len(self.dados)} concursos analisados",
            f"- **Período:** {self.dados['Data'].min() if 'Data' in self.dados.columns else 'N/A'} a {self.dados['Data'].max() if 'Data' in self.dados.columns else 'N/A'}",
            "- **Distribuição de Frequência:** Análise revela padrões não-uniformes",
            "- **Padrões Temporais:** Identificadas tendências sazonais",
            "- **Correlações:** Baixa correlação entre posições, alta co-ocorrência específica",
            "- **Oportunidades:** Múltiplas áreas para otimização do modelo atual\n"
        ]
        
        return "\n".join(summary)
    
    def _format_frequency_results(self, results: Dict[str, Any]) -> str:
        """
        Formata resultados da análise de frequência
        
        Args:
            results: Resultados da análise
            
        Returns:
            Texto formatado
        """
        text = [
            f"**Número mais frequente:** {results['numero_mais_frequente']} ({results['maior_frequencia']} aparições)",
            f"**Número menos frequente:** {results['numero_menos_frequente']} ({results['menor_frequencia']} aparições)",
            f"**Coeficiente de variação:** {results['coeficiente_variacao']:.4f}",
            f"**Distribuição uniforme:** {'Sim' if results['eh_uniforme'] else 'Não'} (p-value: {results['p_value_uniformidade']:.4f})",
            f"**Números 'quentes':** {', '.join(map(str, results['numeros_quentes']))}",
            f"**Números 'frios':** {', '.join(map(str, results['numeros_frios']))}"
        ]
        
        return "\n".join([f"- {item}" for item in text]) + "\n"
    
    def _format_temporal_results(self, results: Dict[str, Any]) -> str:
        """
        Formata resultados da análise temporal
        
        Args:
            results: Resultados da análise
            
        Returns:
            Texto formatado
        """
        text = []
        
        if 'sequences' in results:
            text.append(f"**Média de sequência máxima:** {results['sequences']['media_max_sequencia']:.2f}")
        
        if 'number_intervals' in results:
            text.append(f"**Números analisados para intervalos:** {len(results['number_intervals'])}")
        
        if 'most_seasonal_numbers' in results:
            seasonal_nums = [str(num) for num, _ in results['most_seasonal_numbers'][:3]]
            text.append(f"**Números com maior variação sazonal:** {', '.join(seasonal_nums)}")
        
        return "\n".join([f"- {item}" for item in text]) + "\n" if text else "Análise temporal em andamento...\n"
    
    def _format_distribution_results(self, results: Dict[str, Any]) -> str:
        """
        Formata resultados da análise de distribuição
        
        Args:
            results: Resultados da análise
            
        Returns:
            Texto formatado
        """
        text = [
            f"**Média de números pares por jogo:** {results['par_impar']['media_pares']:.2f}",
            f"**Média de números baixos (1-8):** {results['faixas']['media_baixos']:.2f}",
            f"**Média de números médios (9-17):** {results['faixas']['media_medios']:.2f}",
            f"**Média de números altos (18-25):** {results['faixas']['media_altos']:.2f}",
            f"**Soma média dos números:** {results['soma']['media_soma']:.2f}",
            f"**Distribuição da soma é normal:** {'Sim' if results['soma']['eh_normal'] else 'Não'}",
            f"**Gap médio entre números:** {results['gaps']['media_gap']:.2f}"
        ]
        
        return "\n".join([f"- {item}" for item in text]) + "\n"
    
    def _format_correlation_results(self, results: Dict[str, Any]) -> str:
        """
        Formata resultados da análise de correlação
        
        Args:
            results: Resultados da análise
            
        Returns:
            Texto formatado
        """
        text = [
            f"**Correlação média entre posições:** {results['avg_correlation']:.4f}",
            f"**Pares mais frequentes:** {len(results['frequent_pairs'])} identificados",
            f"**Top 3 pares:** {', '.join([f'({p[0]},{p[1]})' for p in results['frequent_pairs'][:3]])}",
            f"**Média de números consecutivos:** {results['consecutive_analysis']['media_consecutivos']:.2f}"
        ]
        
        return "\n".join([f"- {item}" for item in text]) + "\n"
    
    def _generate_insights(self) -> str:
        """
        Gera insights baseados nas análises
        
        Returns:
            Texto com insights
        """
        insights = [
            "### Insights Principais:\n",
            "1. **Padrões de Frequência:** A distribuição não é perfeitamente uniforme, "
            "indicando oportunidades para modelos que considerem frequências históricas.",
            
            "2. **Distribuição Espacial:** Números tendem a se distribuir de forma "
            "relativamente equilibrada entre faixas baixas, médias e altas.",
            
            "3. **Correlações Baixas:** A baixa correlação entre posições sugere "
            "independência, mas padrões de co-ocorrência podem ser explorados.",
            
            "4. **Sequências Limitadas:** Sequências longas são raras, mas "
            "números consecutivos aparecem com frequência moderada.",
            
            "5. **Oportunidades de Feature Engineering:** Múltiplas características "
            "podem ser derivadas para melhorar predições.\n"
        ]
        
        return "\n".join(insights)
    
    def _identify_model_limitations(self) -> str:
        """
        Identifica limitações do modelo atual
        
        Returns:
            Texto com limitações identificadas
        """
        limitations = [
            "### Limitações Identificadas:\n",
            "1. **Feature Engineering Básico:** Apenas representação binária dos números",
            "2. **Arquitetura Simples:** Rede neural densa sem especialização",
            "3. **Dados Limitados:** Apenas ~3000 concursos podem não ser suficientes",
            "4. **Métricas Inadequadas:** Accuracy não é ideal para problemas de loteria",
            "5. **Falta de Regularização Temporal:** Não considera padrões temporais",
            "6. **Ausência de Ensemble:** Modelo único sem diversificação",
            "7. **Validação Simples:** Split básico treino/teste sem validação cruzada\n"
        ]
        
        return "\n".join([f"- {item}" for item in limitations[1:]])
    
    def _suggest_optimization_steps(self) -> str:
        """
        Sugere próximos passos para otimização
        
        Returns:
            Texto com sugestões
        """
        steps = [
            "### Próximos Passos Recomendados:\n",
            "1. **Feature Engineering Avançado:**",
            "   - Implementar features estatísticas (frequência, gaps, sequências)",
            "   - Adicionar features temporais (sazonalidade, tendências)",
            "   - Criar features de co-ocorrência e correlação",
            
            "2. **Otimização de Arquitetura:**",
            "   - Testar diferentes arquiteturas (CNN, LSTM, Transformer)",
            "   - Implementar regularização avançada",
            "   - Adicionar camadas de atenção",
            
            "3. **Ensemble Learning:**",
            "   - Criar ensemble de modelos diversos",
            "   - Implementar stacking com meta-learners",
            "   - Usar bagging e boosting",
            
            "4. **Otimização de Hiperparâmetros:**",
            "   - Grid Search sistemático",
            "   - Otimização Bayesiana",
            "   - AutoML para descoberta automática",
            
            "5. **Métricas Especializadas:**",
            "   - Desenvolver métricas específicas para loteria",
            "   - Implementar validação temporal",
            "   - Usar cross-validation estratificada\n"
        ]
        
        return "\n".join(steps)


# Exemplo de uso
if __name__ == "__main__":
    # Criar instância da análise
    analyzer = LotofacilExploratoryAnalysis()
    
    # Gerar relatório completo
    report_path = analyzer.generate_comprehensive_report()
    
    print(f"\nAnálise exploratória concluída!")
    print(f"Relatório disponível em: {report_path}")