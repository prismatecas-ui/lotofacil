#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Métricas Avançadas - Lotofácil

Este módulo implementa métricas específicas para avaliação de modelos
de predição de loteria, incluindo métricas de domínio específico,
validação estatística e análise de performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class LotteryMetrics:
    """
    Classe para métricas específicas de loteria
    """
    
    def __init__(self, output_dir: str = "experimentos/resultados"):
        """
        Inicializa o sistema de métricas
        
        Args:
            output_dir: Diretório para salvar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        
    def hit_rate_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         k_values: List[int] = [11, 12, 13, 14, 15]) -> Dict[str, float]:
        """
        Calcula taxa de acerto para diferentes quantidades de números
        
        Args:
            y_true: Números reais sorteados (formato binário 25 posições)
            y_pred: Probabilidades preditas para cada número
            k_values: Lista de quantidades de números para avaliar
            
        Returns:
            Dicionário com taxas de acerto
        """
        hit_rates = {}
        
        for k in k_values:
            hits = 0
            total_games = len(y_true)
            
            for i in range(total_games):
                # Selecionar top-k números com maior probabilidade
                top_k_indices = np.argsort(y_pred[i])[-k:]
                true_numbers = np.where(y_true[i] == 1)[0]
                
                # Calcular interseção
                intersection = len(set(top_k_indices) & set(true_numbers))
                
                # Considerar acerto se pelo menos 11 números coincidirem
                if intersection >= 11:
                    hits += 1
            
            hit_rates[f'hit_rate_{k}'] = hits / total_games if total_games > 0 else 0
        
        return hit_rates
    
    def probability_calibration_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Avalia calibração das probabilidades preditas
        
        Args:
            y_true: Valores reais (formato binário)
            y_pred: Probabilidades preditas
            
        Returns:
            Métricas de calibração
        """
        calibration_metrics = {}
        
        # Flatten arrays para análise por número individual
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Brier Score (menor é melhor)
        brier_score = brier_score_loss(y_true_flat, y_pred_flat)
        calibration_metrics['brier_score'] = brier_score
        
        # Log Loss (menor é melhor)
        # Evitar log(0) adicionando pequeno epsilon
        y_pred_clipped = np.clip(y_pred_flat, 1e-15, 1 - 1e-15)
        log_loss_score = log_loss(y_true_flat, y_pred_clipped)
        calibration_metrics['log_loss'] = log_loss_score
        
        # Reliability (calibração)
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_flat, y_pred_flat, n_bins=10
            )
            
            # Calcular Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_flat > bin_lower) & (y_pred_flat <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true_flat[in_bin].mean()
                    avg_confidence_in_bin = y_pred_flat[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_metrics['expected_calibration_error'] = ece
            
        except Exception as e:
            print(f"Erro no cálculo de calibração: {e}")
            calibration_metrics['expected_calibration_error'] = np.nan
        
        return calibration_metrics
    
    def frequency_based_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               historical_frequencies: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Métricas baseadas em frequência histórica dos números
        
        Args:
            y_true: Números reais sorteados
            y_pred: Probabilidades preditas
            historical_frequencies: Frequências históricas dos números 1-25
            
        Returns:
            Métricas baseadas em frequência
        """
        freq_metrics = {}
        
        # Se não fornecidas, calcular frequências dos dados atuais
        if historical_frequencies is None:
            historical_frequencies = y_true.mean(axis=0)
        
        # Correlação entre probabilidades preditas e frequências históricas
        pred_frequencies = y_pred.mean(axis=0)
        freq_correlation = np.corrcoef(pred_frequencies, historical_frequencies)[0, 1]
        freq_metrics['frequency_correlation'] = freq_correlation
        
        # Desvio médio das frequências preditas vs históricas
        freq_deviation = np.mean(np.abs(pred_frequencies - historical_frequencies))
        freq_metrics['frequency_deviation'] = freq_deviation
        
        # Análise de números quentes vs frios
        hot_threshold = np.percentile(historical_frequencies, 75)
        cold_threshold = np.percentile(historical_frequencies, 25)
        
        hot_numbers = historical_frequencies >= hot_threshold
        cold_numbers = historical_frequencies <= cold_threshold
        
        # Precisão em números quentes
        if np.any(hot_numbers):
            hot_precision = np.mean(y_pred[:, hot_numbers] >= 0.5)
            freq_metrics['hot_numbers_precision'] = hot_precision
        
        # Precisão em números frios
        if np.any(cold_numbers):
            cold_precision = np.mean(y_pred[:, cold_numbers] < 0.5)
            freq_metrics['cold_numbers_precision'] = cold_precision
        
        return freq_metrics
    
    def pattern_recognition_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Métricas para reconhecimento de padrões específicos
        
        Args:
            y_true: Números reais sorteados
            y_pred: Probabilidades preditas
            
        Returns:
            Métricas de padrões
        """
        pattern_metrics = {}
        
        # Análise de distribuição par/ímpar
        odd_positions = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]  # números ímpares (1,3,5...25)
        even_positions = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]     # números pares (2,4,6...24)
        
        # Proporção real de números pares/ímpares
        true_odd_ratio = y_true[:, odd_positions].sum(axis=1).mean() / 15
        true_even_ratio = y_true[:, even_positions].sum(axis=1).mean() / 15
        
        # Proporção predita de números pares/ímpares
        pred_odd_ratio = y_pred[:, odd_positions].sum(axis=1).mean() / 15
        pred_even_ratio = y_pred[:, even_positions].sum(axis=1).mean() / 15
        
        pattern_metrics['odd_even_accuracy'] = 1 - abs(true_odd_ratio - pred_odd_ratio)
        
        # Análise de distribuição por dezenas
        dezena1 = list(range(0, 10))   # números 1-10
        dezena2 = list(range(10, 20))  # números 11-20
        dezena3 = list(range(20, 25))  # números 21-25
        
        for i, (dezena, name) in enumerate([(dezena1, 'dezena1'), (dezena2, 'dezena2'), (dezena3, 'dezena3')]):
            if len(dezena) > 0:
                true_dezena_ratio = y_true[:, dezena].sum(axis=1).mean() / 15
                pred_dezena_ratio = y_pred[:, dezena].sum(axis=1).mean() / 15
                pattern_metrics[f'{name}_accuracy'] = 1 - abs(true_dezena_ratio - pred_dezena_ratio)
        
        # Análise de sequências consecutivas
        consecutive_patterns = self._analyze_consecutive_patterns(y_true, y_pred)
        pattern_metrics.update(consecutive_patterns)
        
        return pattern_metrics
    
    def _analyze_consecutive_patterns(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Analisa padrões de números consecutivos
        
        Args:
            y_true: Números reais
            y_pred: Probabilidades preditas
            
        Returns:
            Métricas de consecutividade
        """
        consecutive_metrics = {}
        
        # Contar sequências consecutivas reais
        true_consecutive_counts = []
        pred_consecutive_counts = []
        
        for i in range(len(y_true)):
            # Números sorteados (posições com 1)
            true_numbers = np.where(y_true[i] == 1)[0] + 1  # +1 para números 1-25
            
            # Top 15 números preditos
            top_15_indices = np.argsort(y_pred[i])[-15:]
            pred_numbers = top_15_indices + 1  # +1 para números 1-25
            
            # Contar consecutivos
            true_consecutive = self._count_consecutive_numbers(sorted(true_numbers))
            pred_consecutive = self._count_consecutive_numbers(sorted(pred_numbers))
            
            true_consecutive_counts.append(true_consecutive)
            pred_consecutive_counts.append(pred_consecutive)
        
        # Métricas de consecutividade
        consecutive_metrics['avg_consecutive_true'] = np.mean(true_consecutive_counts)
        consecutive_metrics['avg_consecutive_pred'] = np.mean(pred_consecutive_counts)
        consecutive_metrics['consecutive_accuracy'] = 1 - abs(
            np.mean(true_consecutive_counts) - np.mean(pred_consecutive_counts)
        )
        
        return consecutive_metrics
    
    def _count_consecutive_numbers(self, numbers: List[int]) -> int:
        """
        Conta quantidade de números consecutivos em uma lista
        
        Args:
            numbers: Lista de números ordenados
            
        Returns:
            Quantidade de números consecutivos
        """
        if len(numbers) < 2:
            return 0
        
        consecutive_count = 0
        current_sequence = 1
        
        for i in range(1, len(numbers)):
            if numbers[i] == numbers[i-1] + 1:
                current_sequence += 1
            else:
                if current_sequence >= 2:
                    consecutive_count += current_sequence
                current_sequence = 1
        
        # Verificar última sequência
        if current_sequence >= 2:
            consecutive_count += current_sequence
        
        return consecutive_count
    
    def statistical_significance_tests(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     baseline_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Testes de significância estatística
        
        Args:
            y_true: Valores reais
            y_pred: Predições do modelo
            baseline_pred: Predições de baseline (opcional)
            
        Returns:
            Resultados dos testes estatísticos
        """
        stat_tests = {}
        
        # Flatten para análise
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Teste de normalidade das predições
        shapiro_stat, shapiro_p = stats.shapiro(y_pred_flat[:5000])  # Limite para performance
        stat_tests['normality_test'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }
        
        # Teste de uniformidade das predições
        ks_stat, ks_p = ks_2samp(y_pred_flat, np.random.uniform(0, 1, len(y_pred_flat)))
        stat_tests['uniformity_test'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_uniform': ks_p > 0.05
        }
        
        # Se baseline fornecido, comparar performances
        if baseline_pred is not None:
            baseline_flat = baseline_pred.flatten()
            
            # Teste de Wilcoxon para comparar distribuições
            try:
                wilcoxon_stat, wilcoxon_p = mannwhitneyu(
                    np.abs(y_true_flat - y_pred_flat),
                    np.abs(y_true_flat - baseline_flat),
                    alternative='two-sided'
                )
                
                stat_tests['model_vs_baseline'] = {
                    'statistic': wilcoxon_stat,
                    'p_value': wilcoxon_p,
                    'model_better': wilcoxon_p < 0.05 and np.mean(np.abs(y_true_flat - y_pred_flat)) < np.mean(np.abs(y_true_flat - baseline_flat))
                }
            except Exception as e:
                stat_tests['model_vs_baseline'] = {'error': str(e)}
        
        # Teste de independência (Chi-quadrado) para números
        try:
            # Criar tabela de contingência para primeiros 1000 jogos
            sample_size = min(1000, len(y_true))
            contingency_table = np.zeros((2, 25))
            
            for i in range(sample_size):
                top_15 = np.argsort(y_pred[i])[-15:]
                for j in range(25):
                    if j in top_15:
                        contingency_table[1, j] += 1
                    else:
                        contingency_table[0, j] += 1
            
            chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
            stat_tests['independence_test'] = {
                'statistic': chi2_stat,
                'p_value': chi2_p,
                'degrees_freedom': dof,
                'is_independent': chi2_p > 0.05
            }
        except Exception as e:
            stat_tests['independence_test'] = {'error': str(e)}
        
        return stat_tests
    
    def roi_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    bet_cost: float = 3.0, prizes: Dict[int, float] = None) -> Dict[str, float]:
        """
        Análise de Retorno sobre Investimento (ROI)
        
        Args:
            y_true: Números reais sorteados
            y_pred: Probabilidades preditas
            bet_cost: Custo de uma aposta
            prizes: Dicionário com prêmios por quantidade de acertos
            
        Returns:
            Métricas de ROI
        """
        if prizes is None:
            # Prêmios aproximados da Lotofácil
            prizes = {
                15: 1500000,  # 15 acertos
                14: 1500,     # 14 acertos
                13: 25,       # 13 acertos
                12: 10,       # 12 acertos
                11: 5         # 11 acertos
            }
        
        roi_metrics = {}
        total_investment = 0
        total_return = 0
        
        for i in range(len(y_true)):
            # Selecionar 15 números com maior probabilidade
            top_15_indices = np.argsort(y_pred[i])[-15:]
            true_numbers = np.where(y_true[i] == 1)[0]
            
            # Calcular acertos
            hits = len(set(top_15_indices) & set(true_numbers))
            
            # Investimento
            total_investment += bet_cost
            
            # Retorno
            if hits in prizes:
                total_return += prizes[hits]
        
        # Calcular métricas de ROI
        roi_metrics['total_investment'] = total_investment
        roi_metrics['total_return'] = total_return
        roi_metrics['net_profit'] = total_return - total_investment
        roi_metrics['roi_percentage'] = ((total_return - total_investment) / total_investment * 100) if total_investment > 0 else 0
        roi_metrics['break_even_rate'] = (total_investment / len(y_true)) / bet_cost if len(y_true) > 0 else 0
        
        # Análise por faixa de acertos
        hit_distribution = {}
        for hits in range(11, 16):
            count = 0
            for i in range(len(y_true)):
                top_15_indices = np.argsort(y_pred[i])[-15:]
                true_numbers = np.where(y_true[i] == 1)[0]
                actual_hits = len(set(top_15_indices) & set(true_numbers))
                if actual_hits == hits:
                    count += 1
            
            hit_distribution[f'hits_{hits}'] = count
            hit_distribution[f'hits_{hits}_rate'] = count / len(y_true) if len(y_true) > 0 else 0
        
        roi_metrics['hit_distribution'] = hit_distribution
        
        return roi_metrics
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               baseline_pred: Optional[np.ndarray] = None,
                               historical_frequencies: Optional[np.ndarray] = None,
                               model_name: str = "Model") -> Dict[str, Any]:
        """
        Avaliação completa do modelo com todas as métricas
        
        Args:
            y_true: Valores reais
            y_pred: Predições do modelo
            baseline_pred: Predições de baseline
            historical_frequencies: Frequências históricas
            model_name: Nome do modelo
            
        Returns:
            Dicionário completo com todas as métricas
        """
        print(f"\n=== AVALIAÇÃO COMPLETA: {model_name} ===")
        
        evaluation_results = {
            'model_name': model_name,
            'evaluation_date': datetime.now().isoformat(),
            'data_shape': {
                'samples': len(y_true),
                'features': y_true.shape[1] if len(y_true.shape) > 1 else 1
            }
        }
        
        # 1. Métricas básicas
        print("Calculando métricas básicas...")
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        basic_metrics = {
            'accuracy': accuracy_score(y_true.flatten(), y_pred_binary.flatten()),
            'precision': precision_score(y_true.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0),
            'recall': recall_score(y_true.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0),
            'f1_score': f1_score(y_true.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
        }
        evaluation_results['basic_metrics'] = basic_metrics
        
        # 2. Métricas específicas de loteria
        print("Calculando métricas de hit rate...")
        hit_rates = self.hit_rate_analysis(y_true, y_pred)
        evaluation_results['hit_rates'] = hit_rates
        
        # 3. Calibração de probabilidades
        print("Analisando calibração de probabilidades...")
        calibration = self.probability_calibration_score(y_true, y_pred)
        evaluation_results['calibration'] = calibration
        
        # 4. Métricas baseadas em frequência
        print("Calculando métricas de frequência...")
        frequency_metrics = self.frequency_based_metrics(y_true, y_pred, historical_frequencies)
        evaluation_results['frequency_metrics'] = frequency_metrics
        
        # 5. Reconhecimento de padrões
        print("Analisando padrões...")
        pattern_metrics = self.pattern_recognition_metrics(y_true, y_pred)
        evaluation_results['pattern_metrics'] = pattern_metrics
        
        # 6. Testes estatísticos
        print("Executando testes estatísticos...")
        statistical_tests = self.statistical_significance_tests(y_true, y_pred, baseline_pred)
        evaluation_results['statistical_tests'] = statistical_tests
        
        # 7. Análise de ROI
        print("Calculando ROI...")
        roi_analysis = self.roi_analysis(y_true, y_pred)
        evaluation_results['roi_analysis'] = roi_analysis
        
        # 8. Score geral
        overall_score = self._calculate_overall_score(evaluation_results)
        evaluation_results['overall_score'] = overall_score
        
        # Salvar resultados
        self._save_evaluation_results(evaluation_results)
        
        # Adicionar ao histórico
        self.metrics_history.append(evaluation_results)
        
        print(f"Avaliação completa finalizada. Score geral: {overall_score:.3f}")
        
        return evaluation_results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calcula score geral baseado em todas as métricas
        
        Args:
            results: Resultados da avaliação
            
        Returns:
            Score geral (0-1)
        """
        scores = []
        weights = []
        
        # Hit rate (peso alto)
        if 'hit_rates' in results:
            hit_15 = results['hit_rates'].get('hit_rate_15', 0)
            scores.append(hit_15)
            weights.append(0.3)
        
        # Calibração (peso médio)
        if 'calibration' in results:
            # Brier score invertido (menor é melhor)
            brier = results['calibration'].get('brier_score', 1)
            calibration_score = max(0, 1 - brier)
            scores.append(calibration_score)
            weights.append(0.2)
        
        # ROI (peso alto)
        if 'roi_analysis' in results:
            roi_pct = results['roi_analysis'].get('roi_percentage', -100)
            roi_score = max(0, min(1, (roi_pct + 100) / 200))  # Normalizar -100% a +100%
            scores.append(roi_score)
            weights.append(0.25)
        
        # Padrões (peso médio)
        if 'pattern_metrics' in results:
            pattern_scores = []
            for key, value in results['pattern_metrics'].items():
                if 'accuracy' in key and isinstance(value, (int, float)):
                    pattern_scores.append(value)
            
            if pattern_scores:
                avg_pattern = np.mean(pattern_scores)
                scores.append(avg_pattern)
                weights.append(0.15)
        
        # Métricas básicas (peso baixo)
        if 'basic_metrics' in results:
            f1 = results['basic_metrics'].get('f1_score', 0)
            scores.append(f1)
            weights.append(0.1)
        
        # Calcular média ponderada
        if scores and weights:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
        
        return float(overall_score)
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """
        Salva resultados da avaliação
        
        Args:
            results: Resultados da avaliação
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = results.get('model_name', 'model').replace(' ', '_')
        
        # Salvar JSON
        json_path = self.output_dir / f"metricas_{model_name}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Métricas salvas em: {json_path}")
    
    def compare_models(self, evaluations: List[Dict[str, Any]]) -> str:
        """
        Compara múltiplos modelos
        
        Args:
            evaluations: Lista de avaliações de modelos
            
        Returns:
            Caminho do relatório de comparação
        """
        print("\n=== COMPARAÇÃO DE MODELOS ===")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"comparacao_modelos_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comparação de Modelos - Lotofácil\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"**Modelos Comparados:** {len(evaluations)}\n\n")
            
            # Tabela de scores gerais
            f.write("## Ranking Geral\n\n")
            f.write("| Posição | Modelo | Score Geral | Hit Rate 15 | ROI (%) | Calibração |\n")
            f.write("|---------|--------|-------------|-------------|---------|------------|\n")
            
            # Ordenar por score geral
            sorted_evals = sorted(evaluations, key=lambda x: x.get('overall_score', 0), reverse=True)
            
            for i, eval_result in enumerate(sorted_evals, 1):
                model_name = eval_result.get('model_name', 'Unknown')
                overall_score = eval_result.get('overall_score', 0)
                hit_rate_15 = eval_result.get('hit_rates', {}).get('hit_rate_15', 0)
                roi_pct = eval_result.get('roi_analysis', {}).get('roi_percentage', 0)
                brier_score = eval_result.get('calibration', {}).get('brier_score', 1)
                
                f.write(f"| {i} | {model_name} | {overall_score:.3f} | {hit_rate_15:.3f} | {roi_pct:.1f} | {1-brier_score:.3f} |\n")
            
            # Análise detalhada por categoria
            f.write("\n## Análise Detalhada\n\n")
            
            categories = [
                ('hit_rates', 'Taxa de Acerto'),
                ('calibration', 'Calibração'),
                ('roi_analysis', 'ROI'),
                ('pattern_metrics', 'Padrões')
            ]
            
            for category_key, category_name in categories:
                f.write(f"### {category_name}\n\n")
                
                for eval_result in sorted_evals:
                    model_name = eval_result.get('model_name', 'Unknown')
                    f.write(f"**{model_name}:**\n")
                    
                    if category_key in eval_result:
                        category_data = eval_result[category_key]
                        for key, value in category_data.items():
                            if isinstance(value, (int, float)):
                                f.write(f"- {key}: {value:.4f}\n")
                    f.write("\n")
            
            # Recomendações
            f.write("## Recomendações\n\n")
            best_model = sorted_evals[0] if sorted_evals else None
            
            if best_model:
                f.write(f"**Melhor Modelo:** {best_model.get('model_name', 'Unknown')}\n\n")
                f.write("**Pontos Fortes:**\n")
                
                # Identificar pontos fortes
                if best_model.get('hit_rates', {}).get('hit_rate_15', 0) > 0.1:
                    f.write("- Excelente taxa de acerto para 15 números\n")
                
                if best_model.get('roi_analysis', {}).get('roi_percentage', 0) > 0:
                    f.write("- ROI positivo\n")
                
                if best_model.get('calibration', {}).get('brier_score', 1) < 0.25:
                    f.write("- Boa calibração de probabilidades\n")
                
                f.write("\n**Áreas de Melhoria:**\n")
                f.write("- Implementar ensemble learning\n")
                f.write("- Otimizar feature engineering\n")
                f.write("- Melhorar calibração de probabilidades\n")
        
        print(f"Relatório de comparação salvo em: {report_path}")
        return str(report_path)
    
    def generate_metrics_dashboard(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Gera dashboard visual das métricas
        
        Args:
            evaluation_results: Resultados da avaliação
            
        Returns:
            Caminho do arquivo do dashboard
        """
        print("Gerando dashboard de métricas...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Dashboard de Métricas - {evaluation_results.get('model_name', 'Model')}", fontsize=16)
        
        # 1. Hit Rates
        if 'hit_rates' in evaluation_results:
            hit_data = evaluation_results['hit_rates']
            k_values = [int(k.split('_')[-1]) for k in hit_data.keys()]
            hit_values = list(hit_data.values())
            
            axes[0, 0].bar(k_values, hit_values, color='skyblue')
            axes[0, 0].set_title('Taxa de Acerto por Quantidade de Números')
            axes[0, 0].set_xlabel('Quantidade de Números')
            axes[0, 0].set_ylabel('Taxa de Acerto')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROI Analysis
        if 'roi_analysis' in evaluation_results and 'hit_distribution' in evaluation_results['roi_analysis']:
            hit_dist = evaluation_results['roi_analysis']['hit_distribution']
            hits = [int(k.split('_')[1]) for k in hit_dist.keys() if 'rate' not in k]
            counts = [hit_dist[f'hits_{h}'] for h in hits]
            
            axes[0, 1].pie(counts, labels=[f'{h} acertos' for h in hits], autopct='%1.1f%%')
            axes[0, 1].set_title('Distribuição de Acertos')
        
        # 3. Pattern Metrics
        if 'pattern_metrics' in evaluation_results:
            pattern_data = evaluation_results['pattern_metrics']
            pattern_names = [k for k in pattern_data.keys() if 'accuracy' in k]
            pattern_values = [pattern_data[k] for k in pattern_names]
            
            if pattern_names:
                axes[0, 2].barh(pattern_names, pattern_values, color='lightgreen')
                axes[0, 2].set_title('Precisão de Padrões')
                axes[0, 2].set_xlabel('Precisão')
        
        # 4. Calibration
        if 'calibration' in evaluation_results:
            cal_data = evaluation_results['calibration']
            metrics = ['brier_score', 'log_loss', 'expected_calibration_error']
            values = [cal_data.get(m, 0) for m in metrics]
            
            axes[1, 0].bar(metrics, values, color='orange')
            axes[1, 0].set_title('Métricas de Calibração')
            axes[1, 0].set_ylabel('Valor')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Basic Metrics
        if 'basic_metrics' in evaluation_results:
            basic_data = evaluation_results['basic_metrics']
            metrics = list(basic_data.keys())
            values = list(basic_data.values())
            
            axes[1, 1].bar(metrics, values, color='lightcoral')
            axes[1, 1].set_title('Métricas Básicas')
            axes[1, 1].set_ylabel('Valor')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Overall Score
        overall_score = evaluation_results.get('overall_score', 0)
        axes[1, 2].pie([overall_score, 1-overall_score], labels=['Score', 'Restante'], 
                      colors=['gold', 'lightgray'], startangle=90)
        axes[1, 2].set_title(f'Score Geral: {overall_score:.3f}')
        
        plt.tight_layout()
        
        # Salvar dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = evaluation_results.get('model_name', 'model').replace(' ', '_')
        dashboard_path = self.output_dir / f"dashboard_{model_name}_{timestamp}.png"
        
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard salvo em: {dashboard_path}")
        return str(dashboard_path)


# Exemplo de uso
if __name__ == "__main__":
    # Criar instância do sistema de métricas
    metrics_system = LotteryMetrics()
    
    # Gerar dados de exemplo
    np.random.seed(42)
    n_samples = 1000
    
    # Simular dados reais (15 números sorteados de 25)
    y_true = np.zeros((n_samples, 25))
    for i in range(n_samples):
        selected = np.random.choice(25, 15, replace=False)
        y_true[i, selected] = 1
    
    # Simular predições do modelo
    y_pred = np.random.random((n_samples, 25))
    
    # Simular baseline (predições aleatórias)
    baseline_pred = np.random.random((n_samples, 25))
    
    print("=== DEMONSTRAÇÃO DO SISTEMA DE MÉTRICAS AVANÇADAS ===")
    
    # Avaliação completa
    evaluation_results = metrics_system.comprehensive_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        baseline_pred=baseline_pred,
        model_name="Modelo Demonstração"
    )
    
    # Gerar dashboard
    dashboard_path = metrics_system.generate_metrics_dashboard(evaluation_results)
    
    print(f"\nDemonstração concluída!")
    print(f"Dashboard disponível em: {dashboard_path}")
    print(f"Score geral do modelo: {evaluation_results['overall_score']:.3f}")