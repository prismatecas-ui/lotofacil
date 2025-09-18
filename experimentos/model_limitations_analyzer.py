#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisador de Limitações do Modelo - Sistema Lotofácil

Este módulo analisa o modelo atual (modelo_tensorflow2.py) e identifica
limitações específicas, pontos de melhoria e oportunidades de otimização.
"""

import sys
import os
import ast
import inspect
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import importlib.util
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from collections import defaultdict

class ModelLimitationsAnalyzer:
    """
    Classe para análise detalhada das limitações do modelo atual
    """
    
    def __init__(self, model_path: str = "modelo/modelo_tensorflow2.py"):
        """
        Inicializa o analisador de limitações
        
        Args:
            model_path: Caminho para o arquivo do modelo
        """
        self.model_path = model_path
        self.analysis_results = {}
        self.output_dir = Path("experimentos/resultados")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar e analisar o código do modelo
        self._load_model_code()
        
    def _load_model_code(self):
        """
        Carrega e analisa o código do modelo
        """
        try:
            with open(self.model_path, 'r', encoding='utf-8') as f:
                self.model_code = f.read()
            
            # Parse do AST para análise estática
            self.ast_tree = ast.parse(self.model_code)
            
            print(f"Código do modelo carregado: {self.model_path}")
            print(f"Linhas de código: {len(self.model_code.splitlines())}")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.model_code = None
            self.ast_tree = None
    
    def analyze_architecture_limitations(self) -> Dict[str, Any]:
        """
        Analisa limitações da arquitetura do modelo
        
        Returns:
            Dicionário com limitações da arquitetura
        """
        print("\n=== ANÁLISE DE LIMITAÇÕES DA ARQUITETURA ===")
        
        arch_limitations = {
            'architecture_type': 'Dense Neural Network',
            'limitations': [],
            'recommendations': [],
            'complexity_score': 0
        }
        
        # Analisar estrutura da rede
        if self.model_code:
            # Verificar tipos de camadas utilizadas
            layer_types = self._extract_layer_types()
            arch_limitations['layer_types'] = layer_types
            
            # Identificar limitações específicas
            if 'Dense' in layer_types and len(layer_types) == 1:
                arch_limitations['limitations'].append({
                    'type': 'Arquitetura Simples',
                    'description': 'Apenas camadas Dense - não explora padrões espaciais ou temporais',
                    'severity': 'Alta',
                    'impact': 'Perda de informações estruturais dos dados'
                })
                arch_limitations['recommendations'].append({
                    'type': 'Arquitetura',
                    'suggestion': 'Implementar CNN para padrões espaciais ou LSTM para sequências temporais'
                })
            
            # Verificar regularização
            regularization_methods = self._check_regularization()
            arch_limitations['regularization'] = regularization_methods
            
            if not regularization_methods['dropout'] and not regularization_methods['batch_norm']:
                arch_limitations['limitations'].append({
                    'type': 'Regularização Insuficiente',
                    'description': 'Falta de técnicas de regularização adequadas',
                    'severity': 'Média',
                    'impact': 'Possível overfitting'
                })
            
            # Verificar função de ativação
            activation_functions = self._extract_activation_functions()
            arch_limitations['activation_functions'] = activation_functions
            
            if 'relu' in activation_functions and len(set(activation_functions)) == 1:
                arch_limitations['limitations'].append({
                    'type': 'Ativação Limitada',
                    'description': 'Uso exclusivo de ReLU pode causar dead neurons',
                    'severity': 'Baixa',
                    'impact': 'Redução da capacidade de aprendizado'
                })
                arch_limitations['recommendations'].append({
                    'type': 'Ativação',
                    'suggestion': 'Experimentar LeakyReLU, ELU, Swish ou outras funções de ativação'
                })
        
        self.analysis_results['architecture_limitations'] = arch_limitations
        
        print(f"Limitações identificadas: {len(arch_limitations['limitations'])}")
        for limitation in arch_limitations['limitations']:
            print(f"- {limitation['type']}: {limitation['description']}")
        
        return arch_limitations
    
    def analyze_data_processing_limitations(self) -> Dict[str, Any]:
        """
        Analisa limitações no processamento de dados
        
        Returns:
            Dicionário com limitações do processamento de dados
        """
        print("\n=== ANÁLISE DE LIMITAÇÕES NO PROCESSAMENTO DE DADOS ===")
        
        data_limitations = {
            'feature_engineering': [],
            'data_representation': [],
            'preprocessing': [],
            'limitations': [],
            'recommendations': []
        }
        
        if self.model_code:
            # Verificar feature engineering
            feature_methods = self._analyze_feature_engineering()
            data_limitations['feature_engineering'] = feature_methods
            
            if not feature_methods['statistical_features']:
                data_limitations['limitations'].append({
                    'type': 'Feature Engineering Básico',
                    'description': 'Ausência de features estatísticas (frequência, gaps, sequências)',
                    'severity': 'Alta',
                    'impact': 'Perda de informações valiosas dos dados históricos'
                })
                data_limitations['recommendations'].append({
                    'type': 'Features',
                    'suggestion': 'Implementar features de frequência, intervalos, co-ocorrência e padrões temporais'
                })
            
            # Verificar representação dos dados
            data_representation = self._analyze_data_representation()
            data_limitations['data_representation'] = data_representation
            
            if data_representation['encoding_type'] == 'binary_only':
                data_limitations['limitations'].append({
                    'type': 'Representação Limitada',
                    'description': 'Apenas codificação binária - não captura relações numéricas',
                    'severity': 'Média',
                    'impact': 'Perda de informações sobre proximidade e ordem dos números'
                })
                data_limitations['recommendations'].append({
                    'type': 'Representação',
                    'suggestion': 'Adicionar embeddings, codificação ordinal ou features numéricas'
                })
            
            # Verificar normalização e preprocessamento
            preprocessing_methods = self._analyze_preprocessing()
            data_limitations['preprocessing'] = preprocessing_methods
            
            if not preprocessing_methods['normalization']:
                data_limitations['limitations'].append({
                    'type': 'Preprocessamento Inadequado',
                    'description': 'Falta de normalização adequada dos dados',
                    'severity': 'Baixa',
                    'impact': 'Possível instabilidade no treinamento'
                })
        
        self.analysis_results['data_processing_limitations'] = data_limitations
        
        print(f"Limitações de dados identificadas: {len(data_limitations['limitations'])}")
        for limitation in data_limitations['limitations']:
            print(f"- {limitation['type']}: {limitation['description']}")
        
        return data_limitations
    
    def analyze_training_limitations(self) -> Dict[str, Any]:
        """
        Analisa limitações no processo de treinamento
        
        Returns:
            Dicionário com limitações do treinamento
        """
        print("\n=== ANÁLISE DE LIMITAÇÕES DO TREINAMENTO ===")
        
        training_limitations = {
            'optimization': [],
            'validation': [],
            'metrics': [],
            'callbacks': [],
            'limitations': [],
            'recommendations': []
        }
        
        if self.model_code:
            # Analisar otimizador
            optimizer_config = self._analyze_optimizer()
            training_limitations['optimization'] = optimizer_config
            
            if optimizer_config['type'] == 'adam' and not optimizer_config['custom_params']:
                training_limitations['limitations'].append({
                    'type': 'Otimização Básica',
                    'description': 'Uso de Adam com parâmetros padrão - não otimizado para o problema',
                    'severity': 'Média',
                    'impact': 'Convergência subótima'
                })
                training_limitations['recommendations'].append({
                    'type': 'Otimização',
                    'suggestion': 'Implementar grid search para learning rate, beta1, beta2 e epsilon'
                })
            
            # Analisar validação
            validation_strategy = self._analyze_validation_strategy()
            training_limitations['validation'] = validation_strategy
            
            if validation_strategy['type'] == 'simple_split':
                training_limitations['limitations'].append({
                    'type': 'Validação Simples',
                    'description': 'Apenas split treino/teste - sem validação cruzada',
                    'severity': 'Alta',
                    'impact': 'Estimativa não confiável da performance real'
                })
                training_limitations['recommendations'].append({
                    'type': 'Validação',
                    'suggestion': 'Implementar validação cruzada temporal ou estratificada'
                })
            
            # Analisar métricas
            metrics_used = self._analyze_metrics()
            training_limitations['metrics'] = metrics_used
            
            if 'accuracy' in metrics_used and len(metrics_used) == 1:
                training_limitations['limitations'].append({
                    'type': 'Métricas Inadequadas',
                    'description': 'Accuracy não é ideal para problemas de loteria',
                    'severity': 'Alta',
                    'impact': 'Avaliação incorreta da qualidade do modelo'
                })
                training_limitations['recommendations'].append({
                    'type': 'Métricas',
                    'suggestion': 'Implementar métricas específicas: hit rate, probabilidade média, entropia'
                })
            
            # Analisar callbacks
            callbacks_used = self._analyze_callbacks()
            training_limitations['callbacks'] = callbacks_used
            
            if not callbacks_used['early_stopping']:
                training_limitations['limitations'].append({
                    'type': 'Controle de Treinamento',
                    'description': 'Ausência de early stopping adequado',
                    'severity': 'Média',
                    'impact': 'Possível overfitting'
                })
        
        self.analysis_results['training_limitations'] = training_limitations
        
        print(f"Limitações de treinamento identificadas: {len(training_limitations['limitations'])}")
        for limitation in training_limitations['limitations']:
            print(f"- {limitation['type']}: {limitation['description']}")
        
        return training_limitations
    
    def analyze_evaluation_limitations(self) -> Dict[str, Any]:
        """
        Analisa limitações na avaliação do modelo
        
        Returns:
            Dicionário com limitações da avaliação
        """
        print("\n=== ANÁLISE DE LIMITAÇÕES DA AVALIAÇÃO ===")
        
        eval_limitations = {
            'metrics_quality': [],
            'validation_approach': [],
            'performance_analysis': [],
            'limitations': [],
            'recommendations': []
        }
        
        # Analisar qualidade das métricas
        metrics_analysis = self._analyze_evaluation_metrics()
        eval_limitations['metrics_quality'] = metrics_analysis
        
        if not metrics_analysis['domain_specific']:
            eval_limitations['limitations'].append({
                'type': 'Métricas Genéricas',
                'description': 'Uso de métricas genéricas não específicas para loteria',
                'severity': 'Alta',
                'impact': 'Avaliação inadequada da utilidade prática do modelo'
            })
            eval_limitations['recommendations'].append({
                'type': 'Métricas Específicas',
                'suggestion': 'Implementar métricas como hit rate por faixa, ROI esperado, probabilidade calibrada'
            })
        
        # Analisar abordagem de validação
        validation_analysis = self._analyze_validation_robustness()
        eval_limitations['validation_approach'] = validation_analysis
        
        if not validation_analysis['temporal_validation']:
            eval_limitations['limitations'].append({
                'type': 'Validação Temporal Ausente',
                'description': 'Não considera a natureza temporal dos dados de loteria',
                'severity': 'Alta',
                'impact': 'Superestimação da performance em dados futuros'
            })
            eval_limitations['recommendations'].append({
                'type': 'Validação Temporal',
                'suggestion': 'Implementar validação com janela deslizante temporal'
            })
        
        # Analisar análise de performance
        performance_analysis = self._analyze_performance_analysis()
        eval_limitations['performance_analysis'] = performance_analysis
        
        if not performance_analysis['statistical_significance']:
            eval_limitations['limitations'].append({
                'type': 'Análise Estatística Insuficiente',
                'description': 'Falta de testes de significância estatística',
                'severity': 'Média',
                'impact': 'Incerteza sobre a confiabilidade dos resultados'
            })
        
        self.analysis_results['evaluation_limitations'] = eval_limitations
        
        print(f"Limitações de avaliação identificadas: {len(eval_limitations['limitations'])}")
        for limitation in eval_limitations['limitations']:
            print(f"- {limitation['type']}: {limitation['description']}")
        
        return eval_limitations
    
    def analyze_scalability_limitations(self) -> Dict[str, Any]:
        """
        Analisa limitações de escalabilidade e performance
        
        Returns:
            Dicionário com limitações de escalabilidade
        """
        print("\n=== ANÁLISE DE LIMITAÇÕES DE ESCALABILIDADE ===")
        
        scalability_limitations = {
            'computational_efficiency': [],
            'memory_usage': [],
            'model_size': [],
            'inference_speed': [],
            'limitations': [],
            'recommendations': []
        }
        
        # Analisar eficiência computacional
        comp_efficiency = self._analyze_computational_efficiency()
        scalability_limitations['computational_efficiency'] = comp_efficiency
        
        if not comp_efficiency['optimized_operations']:
            scalability_limitations['limitations'].append({
                'type': 'Operações Não Otimizadas',
                'description': 'Código não otimizado para operações em lote',
                'severity': 'Média',
                'impact': 'Lentidão em treinamento e inferência'
            })
            scalability_limitations['recommendations'].append({
                'type': 'Otimização',
                'suggestion': 'Implementar operações vetorizadas e processamento em lote'
            })
        
        # Analisar uso de memória
        memory_analysis = self._analyze_memory_usage()
        scalability_limitations['memory_usage'] = memory_analysis
        
        # Analisar tamanho do modelo
        model_size_analysis = self._analyze_model_size()
        scalability_limitations['model_size'] = model_size_analysis
        
        if model_size_analysis['estimated_size'] > 100:  # MB
            scalability_limitations['limitations'].append({
                'type': 'Modelo Grande',
                'description': 'Modelo pode ser muito grande para deployment eficiente',
                'severity': 'Baixa',
                'impact': 'Dificuldades em deployment e inferência rápida'
            })
        
        self.analysis_results['scalability_limitations'] = scalability_limitations
        
        print(f"Limitações de escalabilidade identificadas: {len(scalability_limitations['limitations'])}")
        for limitation in scalability_limitations['limitations']:
            print(f"- {limitation['type']}: {limitation['description']}")
        
        return scalability_limitations
    
    def _extract_layer_types(self) -> List[str]:
        """
        Extrai tipos de camadas utilizadas no modelo
        
        Returns:
            Lista de tipos de camadas
        """
        layer_types = []
        
        # Buscar por padrões de camadas no código
        layer_patterns = [
            'Dense', 'Conv1D', 'Conv2D', 'LSTM', 'GRU', 'Embedding',
            'Dropout', 'BatchNormalization', 'Attention'
        ]
        
        for pattern in layer_patterns:
            if pattern in self.model_code:
                layer_types.append(pattern)
        
        return layer_types
    
    def _check_regularization(self) -> Dict[str, bool]:
        """
        Verifica métodos de regularização utilizados
        
        Returns:
            Dicionário com métodos de regularização
        """
        return {
            'dropout': 'Dropout' in self.model_code,
            'batch_norm': 'BatchNormalization' in self.model_code,
            'l1_l2': any(reg in self.model_code for reg in ['l1', 'l2', 'regularizers']),
            'early_stopping': 'EarlyStopping' in self.model_code
        }
    
    def _extract_activation_functions(self) -> List[str]:
        """
        Extrai funções de ativação utilizadas
        
        Returns:
            Lista de funções de ativação
        """
        activations = []
        activation_patterns = ['relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu', 'swish']
        
        for pattern in activation_patterns:
            if pattern in self.model_code.lower():
                activations.append(pattern)
        
        return activations
    
    def _analyze_feature_engineering(self) -> Dict[str, bool]:
        """
        Analisa feature engineering implementado
        
        Returns:
            Dicionário com features implementadas
        """
        return {
            'statistical_features': any(feat in self.model_code for feat in ['frequency', 'mean', 'std', 'variance']),
            'temporal_features': any(feat in self.model_code for feat in ['date', 'time', 'temporal', 'seasonal']),
            'interaction_features': 'interaction' in self.model_code or 'cross' in self.model_code,
            'polynomial_features': 'polynomial' in self.model_code,
            'embedding_features': 'Embedding' in self.model_code
        }
    
    def _analyze_data_representation(self) -> Dict[str, Any]:
        """
        Analisa representação dos dados
        
        Returns:
            Dicionário com informações da representação
        """
        representation = {
            'encoding_type': 'binary_only',
            'numerical_features': False,
            'categorical_encoding': False,
            'sequence_modeling': False
        }
        
        if any(method in self.model_code for method in ['LabelEncoder', 'OneHotEncoder', 'get_dummies']):
            representation['categorical_encoding'] = True
        
        if any(method in self.model_code for method in ['StandardScaler', 'MinMaxScaler', 'normalize']):
            representation['numerical_features'] = True
        
        if any(method in self.model_code for method in ['LSTM', 'GRU', 'sequence']):
            representation['sequence_modeling'] = True
            representation['encoding_type'] = 'sequence'
        
        return representation
    
    def _analyze_preprocessing(self) -> Dict[str, bool]:
        """
        Analisa métodos de preprocessamento
        
        Returns:
            Dicionário com métodos de preprocessamento
        """
        return {
            'normalization': any(norm in self.model_code for norm in ['StandardScaler', 'MinMaxScaler', 'normalize']),
            'feature_selection': any(sel in self.model_code for sel in ['SelectKBest', 'feature_selection', 'PCA']),
            'outlier_handling': any(out in self.model_code for out in ['outlier', 'IsolationForest', 'LocalOutlierFactor']),
            'missing_value_handling': any(miss in self.model_code for miss in ['fillna', 'impute', 'missing'])
        }
    
    def _analyze_optimizer(self) -> Dict[str, Any]:
        """
        Analisa configuração do otimizador
        
        Returns:
            Dicionário com configuração do otimizador
        """
        optimizer_config = {
            'type': 'adam',  # padrão
            'custom_params': False,
            'learning_rate_schedule': False
        }
        
        if 'Adam' in self.model_code:
            optimizer_config['type'] = 'adam'
        elif 'SGD' in self.model_code:
            optimizer_config['type'] = 'sgd'
        elif 'RMSprop' in self.model_code:
            optimizer_config['type'] = 'rmsprop'
        
        if any(param in self.model_code for param in ['learning_rate', 'lr', 'beta_1', 'beta_2']):
            optimizer_config['custom_params'] = True
        
        if any(schedule in self.model_code for schedule in ['ReduceLROnPlateau', 'ExponentialDecay', 'schedule']):
            optimizer_config['learning_rate_schedule'] = True
        
        return optimizer_config
    
    def _analyze_validation_strategy(self) -> Dict[str, Any]:
        """
        Analisa estratégia de validação
        
        Returns:
            Dicionário com estratégia de validação
        """
        validation_strategy = {
            'type': 'simple_split',
            'cross_validation': False,
            'temporal_validation': False,
            'stratified': False
        }
        
        if any(cv in self.model_code for cv in ['cross_val', 'KFold', 'StratifiedKFold']):
            validation_strategy['cross_validation'] = True
            validation_strategy['type'] = 'cross_validation'
        
        if any(temp in self.model_code for temp in ['TimeSeriesSplit', 'temporal']):
            validation_strategy['temporal_validation'] = True
            validation_strategy['type'] = 'temporal'
        
        if 'Stratified' in self.model_code:
            validation_strategy['stratified'] = True
        
        return validation_strategy
    
    def _analyze_metrics(self) -> List[str]:
        """
        Analisa métricas utilizadas
        
        Returns:
            Lista de métricas
        """
        metrics = []
        metric_patterns = [
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'roc',
            'mae', 'mse', 'rmse', 'mape', 'hit_rate', 'top_k'
        ]
        
        for metric in metric_patterns:
            if metric in self.model_code.lower():
                metrics.append(metric)
        
        return metrics
    
    def _analyze_callbacks(self) -> Dict[str, bool]:
        """
        Analisa callbacks utilizados
        
        Returns:
            Dicionário com callbacks
        """
        return {
            'early_stopping': 'EarlyStopping' in self.model_code,
            'model_checkpoint': 'ModelCheckpoint' in self.model_code,
            'reduce_lr': 'ReduceLROnPlateau' in self.model_code,
            'tensorboard': 'TensorBoard' in self.model_code,
            'csv_logger': 'CSVLogger' in self.model_code
        }
    
    def _analyze_evaluation_metrics(self) -> Dict[str, bool]:
        """
        Analisa qualidade das métricas de avaliação
        
        Returns:
            Dicionário com análise das métricas
        """
        return {
            'domain_specific': any(metric in self.model_code for metric in ['hit_rate', 'lottery', 'probability']),
            'multiple_metrics': len(self._analyze_metrics()) > 1,
            'statistical_tests': any(test in self.model_code for test in ['ttest', 'chi2', 'anova', 'wilcoxon']),
            'confidence_intervals': 'confidence' in self.model_code or 'interval' in self.model_code
        }
    
    def _analyze_validation_robustness(self) -> Dict[str, bool]:
        """
        Analisa robustez da validação
        
        Returns:
            Dicionário com análise da validação
        """
        return {
            'temporal_validation': any(temp in self.model_code for temp in ['TimeSeriesSplit', 'temporal']),
            'cross_validation': any(cv in self.model_code for cv in ['cross_val', 'KFold']),
            'holdout_validation': 'holdout' in self.model_code or 'validation_split' in self.model_code,
            'bootstrap': 'bootstrap' in self.model_code
        }
    
    def _analyze_performance_analysis(self) -> Dict[str, bool]:
        """
        Analisa análise de performance
        
        Returns:
            Dicionário com análise de performance
        """
        return {
            'statistical_significance': any(test in self.model_code for test in ['ttest', 'chi2', 'pvalue']),
            'error_analysis': any(error in self.model_code for error in ['residual', 'error_analysis', 'confusion']),
            'feature_importance': any(imp in self.model_code for imp in ['feature_importance', 'permutation', 'shap']),
            'learning_curves': 'learning_curve' in self.model_code or 'validation_curve' in self.model_code
        }
    
    def _analyze_computational_efficiency(self) -> Dict[str, bool]:
        """
        Analisa eficiência computacional
        
        Returns:
            Dicionário com análise de eficiência
        """
        return {
            'optimized_operations': any(opt in self.model_code for opt in ['vectorize', 'batch', 'parallel']),
            'gpu_utilization': any(gpu in self.model_code for gpu in ['gpu', 'cuda', 'device']),
            'memory_optimization': any(mem in self.model_code for mem in ['memory', 'cache', 'generator']),
            'jit_compilation': any(jit in self.model_code for jit in ['jit', 'xla', 'compile'])
        }
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Analisa uso de memória
        
        Returns:
            Dicionário com análise de memória
        """
        return {
            'memory_efficient': 'generator' in self.model_code or 'batch_size' in self.model_code,
            'data_loading': any(load in self.model_code for load in ['DataLoader', 'Dataset', 'generator']),
            'memory_monitoring': 'memory' in self.model_code and 'usage' in self.model_code
        }
    
    def _analyze_model_size(self) -> Dict[str, Any]:
        """
        Analisa tamanho do modelo
        
        Returns:
            Dicionário com análise do tamanho
        """
        # Estimativa básica baseada na arquitetura
        estimated_params = 0
        
        # Buscar por definições de camadas Dense
        import re
        dense_matches = re.findall(r'Dense\((\d+)', self.model_code)
        
        if dense_matches:
            # Estimativa simples: assumindo entrada de 25 features
            prev_size = 25
            for match in dense_matches:
                layer_size = int(match)
                estimated_params += prev_size * layer_size
                prev_size = layer_size
        
        estimated_size_mb = estimated_params * 4 / (1024 * 1024)  # 4 bytes por parâmetro
        
        return {
            'estimated_parameters': estimated_params,
            'estimated_size': estimated_size_mb,
            'size_category': 'small' if estimated_size_mb < 10 else 'medium' if estimated_size_mb < 100 else 'large'
        }
    
    def generate_comprehensive_limitations_report(self) -> str:
        """
        Gera relatório completo das limitações do modelo
        
        Returns:
            Caminho do arquivo do relatório
        """
        print("\n=== GERANDO RELATÓRIO DE LIMITAÇÕES ===")
        
        # Executar todas as análises
        arch_results = self.analyze_architecture_limitations()
        data_results = self.analyze_data_processing_limitations()
        training_results = self.analyze_training_limitations()
        eval_results = self.analyze_evaluation_limitations()
        scalability_results = self.analyze_scalability_limitations()
        
        # Calcular score geral de limitações
        total_limitations = (
            len(arch_results['limitations']) +
            len(data_results['limitations']) +
            len(training_results['limitations']) +
            len(eval_results['limitations']) +
            len(scalability_results['limitations'])
        )
        
        # Criar relatório
        report_path = self.output_dir / f"relatorio_limitacoes_modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Limitações do Modelo - Lotofácil\n\n")
            f.write(f"**Data da Análise:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"**Modelo Analisado:** {self.model_path}\n")
            f.write(f"**Total de Limitações Identificadas:** {total_limitations}\n\n")
            
            # Resumo Executivo
            f.write("## Resumo Executivo\n\n")
            f.write(self._create_limitations_summary(total_limitations))
            
            # Limitações por categoria
            f.write("\n## 1. Limitações da Arquitetura\n\n")
            f.write(self._format_limitations_section(arch_results))
            
            f.write("\n## 2. Limitações do Processamento de Dados\n\n")
            f.write(self._format_limitations_section(data_results))
            
            f.write("\n## 3. Limitações do Treinamento\n\n")
            f.write(self._format_limitations_section(training_results))
            
            f.write("\n## 4. Limitações da Avaliação\n\n")
            f.write(self._format_limitations_section(eval_results))
            
            f.write("\n## 5. Limitações de Escalabilidade\n\n")
            f.write(self._format_limitations_section(scalability_results))
            
            # Priorização de melhorias
            f.write("\n## 6. Priorização de Melhorias\n\n")
            f.write(self._create_improvement_priorities())
            
            # Roadmap de otimização
            f.write("\n## 7. Roadmap de Otimização\n\n")
            f.write(self._create_optimization_roadmap())
        
        # Salvar dados da análise em JSON
        json_path = self.output_dir / f"dados_limitacoes_modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Relatório de limitações salvo em: {report_path}")
        print(f"Dados da análise salvos em: {json_path}")
        print(f"Total de limitações identificadas: {total_limitations}")
        
        return str(report_path)
    
    def _create_limitations_summary(self, total_limitations: int) -> str:
        """
        Cria resumo das limitações
        
        Args:
            total_limitations: Total de limitações identificadas
            
        Returns:
            Texto do resumo
        """
        severity_level = "Crítica" if total_limitations > 15 else "Alta" if total_limitations > 10 else "Média" if total_limitations > 5 else "Baixa"
        
        summary = [
            f"### Avaliação Geral: Severidade {severity_level}\n",
            "**Principais Problemas Identificados:**",
            "- Arquitetura simples demais para capturar padrões complexos",
            "- Feature engineering básico perdendo informações valiosas",
            "- Métricas inadequadas para avaliação de modelos de loteria",
            "- Validação insuficiente para garantir robustez",
            "- Falta de otimização específica para o domínio\n",
            "**Impacto Estimado:**",
            "- Redução significativa na capacidade preditiva",
            "- Superestimação da performance real",
            "- Dificuldade em identificar padrões sutis",
            "- Baixa confiabilidade dos resultados\n"
        ]
        
        return "\n".join(summary)
    
    def _format_limitations_section(self, results: Dict[str, Any]) -> str:
        """
        Formata seção de limitações
        
        Args:
            results: Resultados da análise
            
        Returns:
            Texto formatado
        """
        if 'limitations' not in results:
            return "Nenhuma limitação específica identificada nesta categoria.\n"
        
        text = []
        
        for i, limitation in enumerate(results['limitations'], 1):
            text.append(f"### {i}. {limitation['type']}")
            text.append(f"**Descrição:** {limitation['description']}")
            text.append(f"**Severidade:** {limitation['severity']}")
            text.append(f"**Impacto:** {limitation['impact']}\n")
        
        if 'recommendations' in results and results['recommendations']:
            text.append("### Recomendações:")
            for rec in results['recommendations']:
                text.append(f"- **{rec['type']}:** {rec['suggestion']}")
            text.append("")
        
        return "\n".join(text)
    
    def _create_improvement_priorities(self) -> str:
        """
        Cria priorização de melhorias
        
        Returns:
            Texto com prioridades
        """
        priorities = [
            "### Prioridade Alta (Crítica):\n",
            "1. **Implementar Feature Engineering Avançado**",
            "   - Features estatísticas (frequência, gaps, co-ocorrência)",
            "   - Features temporais (sazonalidade, tendências)",
            "   - Features de padrões (sequências, distribuições)\n",
            
            "2. **Desenvolver Métricas Específicas para Loteria**",
            "   - Hit rate por faixa de números",
            "   - Probabilidade calibrada",
            "   - ROI esperado\n",
            
            "3. **Implementar Validação Temporal Robusta**",
            "   - Validação com janela deslizante",
            "   - Cross-validation temporal",
            "   - Testes de significância estatística\n",
            
            "### Prioridade Média (Importante):\n",
            "4. **Otimizar Arquitetura do Modelo**",
            "   - Testar arquiteturas especializadas (CNN, LSTM)",
            "   - Implementar ensemble learning",
            "   - Adicionar regularização avançada\n",
            
            "5. **Melhorar Processo de Treinamento**",
            "   - Grid search para hiperparâmetros",
            "   - Otimização Bayesiana",
            "   - Callbacks avançados\n",
            
            "### Prioridade Baixa (Desejável):\n",
            "6. **Otimizar Performance e Escalabilidade**",
            "   - Operações vetorizadas",
            "   - Utilização de GPU",
            "   - Otimização de memória\n"
        ]
        
        return "\n".join(priorities)
    
    def _create_optimization_roadmap(self) -> str:
        """
        Cria roadmap de otimização
        
        Returns:
            Texto com roadmap
        """
        roadmap = [
            "### Fase 1: Fundação (Semanas 1-2)\n",
            "- [ ] Implementar sistema de feature engineering avançado",
            "- [ ] Desenvolver métricas específicas para loteria",
            "- [ ] Configurar validação temporal robusta",
            "- [ ] Estabelecer baseline com métricas adequadas\n",
            
            "### Fase 2: Otimização (Semanas 3-4)\n",
            "- [ ] Experimentar arquiteturas alternativas (CNN, LSTM, Transformer)",
            "- [ ] Implementar ensemble learning",
            "- [ ] Otimizar hiperparâmetros com grid search/Bayesian optimization",
            "- [ ] Adicionar regularização avançada\n",
            
            "### Fase 3: Refinamento (Semanas 5-6)\n",
            "- [ ] Implementar AutoML para descoberta automática",
            "- [ ] Otimizar performance e escalabilidade",
            "- [ ] Desenvolver sistema de monitoramento contínuo",
            "- [ ] Criar pipeline de deployment automatizado\n",
            
            "### Fase 4: Validação (Semana 7)\n",
            "- [ ] Validação extensiva com dados históricos",
            "- [ ] Testes A/B com diferentes abordagens",
            "- [ ] Análise de robustez e estabilidade",
            "- [ ] Documentação completa e transferência de conhecimento\n",
            
            "### Métricas de Sucesso:\n",
            "- Aumento de pelo menos 20% na taxa de acerto",
            "- Redução de 50% na variância das predições",
            "- Melhoria na calibração de probabilidades",
            "- Validação estatisticamente significativa dos resultados\n"
        ]
        
        return "\n".join(roadmap)


# Exemplo de uso
if __name__ == "__main__":
    # Criar instância do analisador
    analyzer = ModelLimitationsAnalyzer()
    
    # Gerar relatório completo
    report_path = analyzer.generate_comprehensive_limitations_report()
    
    print(f"\nAnálise de limitações concluída!")
    print(f"Relatório disponível em: {report_path}")