import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score,
    cross_validate, validation_curve, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import warnings
from pathlib import Path
import joblib
from scipy import stats
from collections import defaultdict

warnings.filterwarnings('ignore')


class ValidacaoMetricas:
    """
    Sistema completo de validação cruzada e métricas para modelos da Lotofácil
    """
    
    def __init__(self, output_dir: str = "./modelo/validacao_resultados"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.resultados_validacao = {}
        self.metricas_personalizadas = {}
        self.historico_validacoes = []
        
    def preparar_dados_validacao(self, dados: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepara dados especificamente para validação
        """
        
        # Identificar colunas de números
        colunas_numeros = []
        for i in range(1, 16):
            for formato in [f'Bola{i:02d}', f'bola{i:02d}', f'Numero{i:02d}', str(i)]:
                if formato in dados.columns:
                    colunas_numeros.append(formato)
                    break
        
        if len(colunas_numeros) < 15:
            colunas_numericas = dados.select_dtypes(include=[np.number]).columns
            colunas_numeros = colunas_numericas[:15].tolist()
        
        # Criar features e targets
        X = []
        y = []
        metadados = []
        
        for i in range(len(dados) - 1):
            # Features do sorteio atual
            sorteio_atual = dados.iloc[i][colunas_numeros]
            proximo_sorteio = dados.iloc[i + 1][colunas_numeros]
            
            # Vetor binário para números 1-25
            vetor_atual = [0] * 25
            numeros_atual = []
            for num in sorteio_atual:
                if pd.notna(num) and 1 <= int(num) <= 25:
                    vetor_atual[int(num) - 1] = 1
                    numeros_atual.append(int(num))
            
            # Features adicionais
            features_extras = self.extrair_features_extras(numeros_atual)
            vetor_completo = vetor_atual + features_extras
            
            # Target
            vetor_target = [0] * 25
            numeros_target = []
            for num in proximo_sorteio:
                if pd.notna(num) and 1 <= int(num) <= 25:
                    vetor_target[int(num) - 1] = 1
                    numeros_target.append(int(num))
            
            X.append(vetor_completo)
            y.append(vetor_target)
            
            # Metadados para análise
            metadados.append({
                'indice': i,
                'numeros_atual': numeros_atual,
                'numeros_target': numeros_target,
                'soma_atual': sum(numeros_atual),
                'soma_target': sum(numeros_target)
            })
        
        info_dados = {
            'total_amostras': len(X),
            'dimensao_features': len(X[0]) if X else 0,
            'dimensao_target': len(y[0]) if y else 0,
            'metadados': metadados
        }
        
        return np.array(X), np.array(y), info_dados
    
    def extrair_features_extras(self, numeros: List[int]) -> List[float]:
        """
        Extrai features adicionais dos números
        """
        
        if not numeros:
            return [0] * 10
        
        features = []
        
        # Estatísticas básicas
        features.append(np.mean(numeros) / 25)  # Média normalizada
        features.append(np.std(numeros) / 25)   # Desvio padrão normalizado
        features.append(min(numeros) / 25)      # Mínimo normalizado
        features.append(max(numeros) / 25)      # Máximo normalizado
        
        # Paridade
        pares = sum(1 for n in numeros if n % 2 == 0)
        features.append(pares / 15)  # Proporção de pares
        
        # Distribuição por dezenas
        dezenas = [0, 0, 0, 0, 0]  # 1-5, 6-10, 11-15, 16-20, 21-25
        for num in numeros:
            if 1 <= num <= 5:
                dezenas[0] += 1
            elif 6 <= num <= 10:
                dezenas[1] += 1
            elif 11 <= num <= 15:
                dezenas[2] += 1
            elif 16 <= num <= 20:
                dezenas[3] += 1
            elif 21 <= num <= 25:
                dezenas[4] += 1
        
        # Adicionar proporção de cada dezena
        features.extend([d / 15 for d in dezenas])
        
        return features
    
    def validacao_cruzada_temporal(self, X: np.ndarray, y: np.ndarray, modelo, n_splits: int = 5) -> Dict[str, Any]:
        """
        Validação cruzada respeitando ordem temporal
        """
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'custom_lotofacil': []
        }
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Treinar modelo
            if hasattr(modelo, 'fit'):
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
            else:
                # Para modelos TensorFlow
                modelo.fit(X_train, y_train, epochs=50, verbose=0)
                y_pred = modelo.predict(X_test)
            
            # Converter predições para binário
            if y_pred.ndim > 1:
                y_pred_binary = (y_pred > 0.5).astype(int)
            else:
                y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calcular métricas
            fold_metrics = self.calcular_metricas_fold(y_test, y_pred_binary, y_pred)
            
            for metric, value in fold_metrics.items():
                if metric in scores:
                    scores[metric].append(value)
            
            fold_results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': fold_metrics
            })
        
        # Calcular estatísticas finais
        final_scores = {}
        for metric, values in scores.items():
            final_scores[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return {
            'tipo_validacao': 'temporal',
            'n_splits': n_splits,
            'scores': final_scores,
            'fold_results': fold_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def validacao_cruzada_estratificada(self, X: np.ndarray, y: np.ndarray, modelo, n_splits: int = 5) -> Dict[str, Any]:
        """
        Validação cruzada estratificada
        """
        
        # Criar labels estratificadas baseadas na soma dos números
        y_somas = np.sum(y, axis=1)
        y_estratificada = pd.cut(y_somas, bins=5, labels=False)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'custom_lotofacil': []
        }
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_estratificada)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Treinar e avaliar
            if hasattr(modelo, 'fit'):
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
            else:
                modelo.fit(X_train, y_train, epochs=50, verbose=0)
                y_pred = modelo.predict(X_test)
            
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            fold_metrics = self.calcular_metricas_fold(y_test, y_pred_binary, y_pred)
            
            for metric, value in fold_metrics.items():
                if metric in scores:
                    scores[metric].append(value)
            
            fold_results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': fold_metrics
            })
        
        final_scores = {}
        for metric, values in scores.items():
            final_scores[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return {
            'tipo_validacao': 'estratificada',
            'n_splits': n_splits,
            'scores': final_scores,
            'fold_results': fold_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def calcular_metricas_fold(self, y_true: np.ndarray, y_pred_binary: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas para um fold específico
        """
        
        metrics = {}
        
        # Métricas padrão
        metrics['accuracy'] = accuracy_score(y_true.flatten(), y_pred_binary.flatten())
        metrics['precision'] = precision_score(y_true.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
        
        # Métrica personalizada para Lotofácil
        metrics['custom_lotofacil'] = self.metrica_lotofacil(y_true, y_pred_binary, y_pred_prob)
        
        # Métricas por número
        metrics['accuracy_por_numero'] = self.accuracy_por_numero(y_true, y_pred_binary)
        
        return metrics
    
    def metrica_lotofacil(self, y_true: np.ndarray, y_pred_binary: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """
        Métrica personalizada para Lotofácil considerando acertos parciais
        """
        
        scores = []
        
        for i in range(len(y_true)):
            true_numbers = set(np.where(y_true[i] == 1)[0] + 1)
            pred_numbers = set(np.where(y_pred_binary[i] == 1)[0] + 1)
            
            # Acertos (interseção)
            acertos = len(true_numbers.intersection(pred_numbers))
            
            # Score baseado na tabela de premiação da Lotofácil
            if acertos >= 11:
                if acertos == 15:
                    score = 1.0  # Acerto total
                elif acertos == 14:
                    score = 0.8
                elif acertos == 13:
                    score = 0.6
                elif acertos == 12:
                    score = 0.4
                else:  # 11 acertos
                    score = 0.2
            else:
                score = 0.0
            
            scores.append(score)
        
        return np.mean(scores)
    
    def accuracy_por_numero(self, y_true: np.ndarray, y_pred_binary: np.ndarray) -> Dict[int, float]:
        """
        Calcula accuracy para cada número individualmente
        """
        
        accuracy_numeros = {}
        
        for num in range(25):
            true_col = y_true[:, num]
            pred_col = y_pred_binary[:, num]
            accuracy_numeros[num + 1] = accuracy_score(true_col, pred_col)
        
        return accuracy_numeros
    
    def curva_aprendizado(self, X: np.ndarray, y: np.ndarray, modelo, train_sizes: np.ndarray = None) -> Dict[str, Any]:
        """
        Gera curva de aprendizado
        """
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                modelo, X, y,
                train_sizes=train_sizes,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                'val_scores_std': np.std(val_scores, axis=1).tolist()
            }
        except Exception as e:
            return {'erro': str(e)}
    
    def analise_residuos(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Análise de resíduos para modelos de regressão
        """
        
        # Converter para formato de soma para análise
        y_true_somas = np.sum(y_true, axis=1)
        y_pred_somas = np.sum(y_pred, axis=1)
        
        residuos = y_true_somas - y_pred_somas
        
        # Testes estatísticos
        shapiro_stat, shapiro_p = stats.shapiro(residuos)
        
        return {
            'residuos_mean': np.mean(residuos),
            'residuos_std': np.std(residuos),
            'residuos_min': np.min(residuos),
            'residuos_max': np.max(residuos),
            'normalidade_shapiro': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'normal': shapiro_p > 0.05
            },
            'residuos_valores': residuos.tolist()
        }
    
    def matriz_confusao_multiclass(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Matriz de confusão para classificação multi-label
        """
        
        # Análise por número
        matrizes_por_numero = {}
        
        for num in range(25):
            cm = confusion_matrix(y_true[:, num], y_pred[:, num])
            matrizes_por_numero[num + 1] = {
                'matriz': cm.tolist(),
                'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
                'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0
            }
        
        return matrizes_por_numero
    
    def validacao_completa(self, X: np.ndarray, y: np.ndarray, modelo, nome_modelo: str) -> Dict[str, Any]:
        """
        Executa validação completa do modelo
        """
        
        print(f"Iniciando validação completa para {nome_modelo}...")
        
        resultado_completo = {
            'modelo': nome_modelo,
            'timestamp': datetime.now().isoformat(),
            'dados_info': {
                'n_amostras': len(X),
                'n_features': X.shape[1],
                'n_targets': y.shape[1]
            }
        }
        
        # Validação cruzada temporal
        print("Executando validação cruzada temporal...")
        resultado_completo['validacao_temporal'] = self.validacao_cruzada_temporal(X, y, modelo)
        
        # Validação cruzada estratificada
        print("Executando validação cruzada estratificada...")
        resultado_completo['validacao_estratificada'] = self.validacao_cruzada_estratificada(X, y, modelo)
        
        # Curva de aprendizado
        print("Gerando curva de aprendizado...")
        resultado_completo['curva_aprendizado'] = self.curva_aprendizado(X, y, modelo)
        
        # Treinar modelo completo para análises adicionais
        print("Treinando modelo completo...")
        if hasattr(modelo, 'fit'):
            modelo.fit(X, y)
            y_pred = modelo.predict(X)
        else:
            modelo.fit(X, y, epochs=100, verbose=0)
            y_pred = modelo.predict(X)
        
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Análise de resíduos
        resultado_completo['analise_residuos'] = self.analise_residuos(y, y_pred)
        
        # Matriz de confusão
        resultado_completo['matriz_confusao'] = self.matriz_confusao_multiclass(y, y_pred_binary)
        
        # Métricas detalhadas
        resultado_completo['metricas_detalhadas'] = self.calcular_metricas_fold(y, y_pred_binary, y_pred)
        
        # Salvar resultado
        self.salvar_resultado_validacao(resultado_completo, nome_modelo)
        
        print(f"Validação completa para {nome_modelo} concluída.")
        
        return resultado_completo
    
    def comparar_modelos(self, resultados_modelos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compara múltiplos modelos
        """
        
        comparacao = {
            'timestamp': datetime.now().isoformat(),
            'modelos_comparados': [r['modelo'] for r in resultados_modelos],
            'metricas_comparacao': {},
            'ranking': {},
            'recomendacao': ''
        }
        
        # Extrair métricas principais
        metricas_principais = ['accuracy', 'precision', 'recall', 'f1', 'custom_lotofacil']
        
        for metrica in metricas_principais:
            comparacao['metricas_comparacao'][metrica] = {}
            
            for resultado in resultados_modelos:
                nome_modelo = resultado['modelo']
                
                # Pegar da validação temporal
                if 'validacao_temporal' in resultado and metrica in resultado['validacao_temporal']['scores']:
                    valor = resultado['validacao_temporal']['scores'][metrica]['mean']
                    comparacao['metricas_comparacao'][metrica][nome_modelo] = valor
        
        # Criar ranking
        for metrica in metricas_principais:
            if metrica in comparacao['metricas_comparacao']:
                valores = comparacao['metricas_comparacao'][metrica]
                ranking = sorted(valores.items(), key=lambda x: x[1], reverse=True)
                comparacao['ranking'][metrica] = ranking
        
        # Gerar recomendação
        if 'custom_lotofacil' in comparacao['ranking']:
            melhor_modelo = comparacao['ranking']['custom_lotofacil'][0][0]
            melhor_score = comparacao['ranking']['custom_lotofacil'][0][1]
            comparacao['recomendacao'] = f"Modelo recomendado: {melhor_modelo} (Score Lotofácil: {melhor_score:.4f})"
        
        return comparacao
    
    def gerar_relatorio_validacao(self, resultado_validacao: Dict[str, Any]) -> str:
        """
        Gera relatório textual da validação
        """
        
        relatorio = []
        relatorio.append(f"RELATÓRIO DE VALIDAÇÃO - {resultado_validacao['modelo']}")
        relatorio.append("=" * 60)
        relatorio.append(f"Data: {resultado_validacao['timestamp']}")
        relatorio.append(f"Amostras: {resultado_validacao['dados_info']['n_amostras']}")
        relatorio.append(f"Features: {resultado_validacao['dados_info']['n_features']}")
        relatorio.append("")
        
        # Validação temporal
        if 'validacao_temporal' in resultado_validacao:
            vt = resultado_validacao['validacao_temporal']
            relatorio.append("VALIDAÇÃO CRUZADA TEMPORAL:")
            relatorio.append("-" * 30)
            
            for metrica, valores in vt['scores'].items():
                relatorio.append(f"{metrica.upper()}: {valores['mean']:.4f} (±{valores['std']:.4f})")
            relatorio.append("")
        
        # Métricas detalhadas
        if 'metricas_detalhadas' in resultado_validacao:
            md = resultado_validacao['metricas_detalhadas']
            relatorio.append("MÉTRICAS DETALHADAS:")
            relatorio.append("-" * 20)
            relatorio.append(f"Accuracy: {md['accuracy']:.4f}")
            relatorio.append(f"Precision: {md['precision']:.4f}")
            relatorio.append(f"Recall: {md['recall']:.4f}")
            relatorio.append(f"F1-Score: {md['f1']:.4f}")
            relatorio.append(f"Score Lotofácil: {md['custom_lotofacil']:.4f}")
            relatorio.append("")
        
        return "\n".join(relatorio)
    
    def salvar_resultado_validacao(self, resultado: Dict[str, Any], nome_modelo: str):
        """
        Salva resultado da validação
        """
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salvar JSON
        json_file = self.output_dir / f"validacao_{nome_modelo}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False, default=str)
        
        # Salvar relatório texto
        relatorio = self.gerar_relatorio_validacao(resultado)
        txt_file = self.output_dir / f"relatorio_{nome_modelo}_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(relatorio)
        
        # Adicionar ao histórico
        self.historico_validacoes.append({
            'modelo': nome_modelo,
            'timestamp': timestamp,
            'arquivo_json': str(json_file),
            'arquivo_txt': str(txt_file)
        })
        
        print(f"Resultados salvos: {json_file}")


def executar_validacao_completa(dados: pd.DataFrame, modelos: Dict[str, Any]) -> Dict[str, Any]:
    """
    Função principal para executar validação completa
    """
    
    print("Iniciando sistema de validação completa...")
    
    validador = ValidacaoMetricas()
    
    # Preparar dados
    X, y, info_dados = validador.preparar_dados_validacao(dados)
    print(f"Dados preparados: {info_dados['total_amostras']} amostras, {info_dados['dimensao_features']} features")
    
    # Validar cada modelo
    resultados_todos_modelos = []
    
    for nome_modelo, modelo in modelos.items():
        print(f"\nValidando modelo: {nome_modelo}")
        resultado = validador.validacao_completa(X, y, modelo, nome_modelo)
        resultados_todos_modelos.append(resultado)
    
    # Comparar modelos
    if len(resultados_todos_modelos) > 1:
        print("\nComparando modelos...")
        comparacao = validador.comparar_modelos(resultados_todos_modelos)
        
        # Salvar comparação
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comp_file = validador.output_dir / f"comparacao_modelos_{timestamp}.json"
        with open(comp_file, 'w', encoding='utf-8') as f:
            json.dump(comparacao, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Comparação salva: {comp_file}")
        print(f"Recomendação: {comparacao['recomendacao']}")
        
        return {
            'resultados_individuais': resultados_todos_modelos,
            'comparacao': comparacao,
            'validador': validador
        }
    
    return {
        'resultados_individuais': resultados_todos_modelos,
        'validador': validador
    }


if __name__ == "__main__":
    # Exemplo de uso
    print("Sistema de Validação e Métricas para Lotofácil")
    print("Este módulo deve ser importado e usado com dados e modelos reais.")