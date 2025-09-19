#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Otimização Avançada para Modelo Lotofácil
Objetivo: Atingir 85-90% de acurácia com feature engineering avançada
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Imports locais
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dados.dados import carregar_dados
from experimentos.feature_engineering import FeatureEngineeringLotofacil

def formatar_porcentagem(valor):
    """Converte decimal para porcentagem formatada"""
    return f"{valor * 100:.2f}%"

def criar_features_super_avancadas(df):
    """Cria features super avançadas com engenharia complexa"""
    print("   🚀 Criando features super avançadas...")
    
    fe = FeatureEngineeringLotofacil()
    
    # 1. Features estatísticas básicas
    features_estat = fe.criar_features_estatisticas(df)
    print(f"      ✓ Features estatísticas: {features_estat.shape[1]} colunas")
    
    # 2. Features temporais
    features_temp = fe.criar_features_temporais(df)
    print(f"      ✓ Features temporais: {features_temp.shape[1]} colunas")
    
    # 3. Features de padrões
    features_padroes = fe.criar_features_padroes(df)
    print(f"      ✓ Features de padrões: {len(features_padroes)} colunas")
    
    # Converter features_padroes para DataFrame se necessário
    if isinstance(features_padroes, dict):
        features_padroes_df = pd.DataFrame([features_padroes] * len(df))
    else:
        features_padroes_df = features_padroes
    
    # 4. Features avançadas customizadas
    features_custom = criar_features_customizadas(df)
    print(f"      ✓ Features customizadas: {features_custom.shape[1]} colunas")
    
    # Combinar todas as features
    features_list = [features_estat]
    
    if features_temp.shape[1] > 0:
        features_list.append(features_temp)
    
    if features_padroes_df.shape[1] > 0:
        features_list.append(features_padroes_df)
    
    features_list.append(features_custom)
    
    features_completas = pd.concat(features_list, axis=1)
    
    # Remover colunas duplicadas e com variância zero
    features_completas = features_completas.loc[:, ~features_completas.columns.duplicated()]
    features_completas = features_completas.loc[:, features_completas.var() > 0]
    
    print(f"   ✅ Total de features criadas: {features_completas.shape[1]}")
    return features_completas

def criar_features_customizadas(df):
    """Cria features customizadas específicas para Lotofácil"""
    features_custom = pd.DataFrame(index=df.index)
    
    # Extrair números de cada concurso
    numeros_cols = [f'Bola_{i:02d}' for i in range(1, 16)]
    
    for idx, row in df.iterrows():
        numeros = [row[col] for col in numeros_cols if col in row and pd.notna(row[col])]
        
        if len(numeros) == 15:
            # Features de distribuição avançadas
            features_custom.loc[idx, 'soma_total'] = sum(numeros)
            features_custom.loc[idx, 'media_numeros'] = np.mean(numeros)
            features_custom.loc[idx, 'desvio_padrao'] = np.std(numeros)
            features_custom.loc[idx, 'amplitude'] = max(numeros) - min(numeros)
            
            # Features de posição
            features_custom.loc[idx, 'numeros_baixos'] = sum(1 for n in numeros if n <= 12)
            features_custom.loc[idx, 'numeros_altos'] = sum(1 for n in numeros if n >= 14)
            features_custom.loc[idx, 'numeros_medios'] = sum(1 for n in numeros if 13 <= n <= 13)
            
            # Features de paridade
            features_custom.loc[idx, 'pares'] = sum(1 for n in numeros if n % 2 == 0)
            features_custom.loc[idx, 'impares'] = sum(1 for n in numeros if n % 2 == 1)
            
            # Features de sequências
            numeros_sorted = sorted(numeros)
            sequencias = 0
            for i in range(len(numeros_sorted) - 1):
                if numeros_sorted[i+1] - numeros_sorted[i] == 1:
                    sequencias += 1
            features_custom.loc[idx, 'sequencias_consecutivas'] = sequencias
            
            # Features de quadrantes
            q1 = sum(1 for n in numeros if 1 <= n <= 6)
            q2 = sum(1 for n in numeros if 7 <= n <= 12)
            q3 = sum(1 for n in numeros if 13 <= n <= 18)
            q4 = sum(1 for n in numeros if 19 <= n <= 25)
            
            features_custom.loc[idx, 'quadrante_1'] = q1
            features_custom.loc[idx, 'quadrante_2'] = q2
            features_custom.loc[idx, 'quadrante_3'] = q3
            features_custom.loc[idx, 'quadrante_4'] = q4
            
            # Features de distância
            distancias = []
            for i in range(len(numeros_sorted) - 1):
                distancias.append(numeros_sorted[i+1] - numeros_sorted[i])
            
            if distancias:
                features_custom.loc[idx, 'distancia_media'] = np.mean(distancias)
                features_custom.loc[idx, 'distancia_max'] = max(distancias)
                features_custom.loc[idx, 'distancia_min'] = min(distancias)
        else:
            # Valores padrão para casos com dados incompletos
            for col in ['soma_total', 'media_numeros', 'desvio_padrao', 'amplitude',
                       'numeros_baixos', 'numeros_altos', 'numeros_medios',
                       'pares', 'impares', 'sequencias_consecutivas',
                       'quadrante_1', 'quadrante_2', 'quadrante_3', 'quadrante_4',
                       'distancia_media', 'distancia_max', 'distancia_min']:
                features_custom.loc[idx, col] = 0
    
    # Preencher valores NaN
    features_custom = features_custom.fillna(0)
    
    return features_custom

def criar_ensemble_super_otimizado(X_train, y_train):
    """Cria ensemble super otimizado com hyperparameter tuning"""
    print("   🎯 Criando ensemble super otimizado...")
    
    # Definir modelos base com parâmetros otimizados
    modelos = {
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'lr': LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            learning_rate_init=0.01,
            max_iter=500,
            random_state=42
        )
    }
    
    # Treinar modelos individuais
    modelos_treinados = {}
    for nome, modelo in modelos.items():
        print(f"      🔄 Treinando {nome.upper()}...")
        modelo.fit(X_train, y_train)
        modelos_treinados[nome] = modelo
    
    # Criar ensemble votante
    ensemble = VotingClassifier(
        estimators=[(nome, modelo) for nome, modelo in modelos_treinados.items()],
        voting='soft'
    )
    
    print("      🔄 Treinando ensemble final...")
    ensemble.fit(X_train, y_train)
    
    return ensemble, modelos_treinados

def aplicar_balanceamento_avancado(X_train, y_train):
    """Aplica balanceamento avançado dos dados"""
    print("   ⚖️ Aplicando balanceamento avançado...")
    
    # Verificar distribuição original
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"      📊 Distribuição original: {dict(zip(unique, counts))}")
    
    # Aplicar SMOTE para oversampling da classe minoritária
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    # Verificar nova distribuição
    unique, counts = np.unique(y_balanced, return_counts=True)
    print(f"      📊 Distribuição balanceada: {dict(zip(unique, counts))}")
    
    return X_balanced, y_balanced

def selecionar_features_otimas(X_train, y_train, k=50):
    """Seleciona as melhores features usando métodos estatísticos"""
    print(f"   🎯 Selecionando {k} melhores features...")
    
    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    X_selected = selector.fit_transform(X_train, y_train)
    
    # Obter nomes das features selecionadas
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    
    print(f"      ✓ Features selecionadas: {len(selected_features)}")
    
    return X_selected, selector, selected_features

def avaliar_modelo_completo(modelo, X_test, y_test):
    """Avalia o modelo com métricas completas"""
    print("   📊 Avaliando modelo...")
    
    # Predições
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC (se possível)
    try:
        if hasattr(modelo, 'predict_proba'):
            y_proba = modelo.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = 0.5
    except:
        roc_auc = 0.5
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def main():
    """Função principal de otimização avançada"""
    print("🚀 INICIANDO OTIMIZAÇÃO AVANÇADA DO MODELO LOTOFÁCIL")
    print("=" * 60)
    
    # 1. Carregar dados
    print("\n1. Carregando dados...")
    df_dados = carregar_dados()
    print(f"   ✓ Dados carregados: {len(df_dados)} concursos")
    
    # 2. Criar features super avançadas
    print("\n2. Criando features super avançadas...")
    features_df = criar_features_super_avancadas(df_dados)
    print(f"   ✓ Features criadas: {features_df.shape}")
    
    # 3. Preparar targets
    print("\n3. Preparando targets...")
    targets = []
    for idx, row in df_dados.iterrows():
        ganhou = 1 if row.get('Ganhadores_Sena', 0) > 0 else 0
        targets.append(ganhou)
    
    y = np.array(targets)
    print(f"   ✓ Targets preparados: {len(y)} amostras")
    
    # Verificar distribuição
    unique, counts = np.unique(y, return_counts=True)
    print(f"   ✓ Distribuição - {dict(zip(unique, counts))}")
    
    # 4. Remover coluna target das features se existir
    if 'Ganhou' in features_df.columns:
        features_df = features_df.drop('Ganhou', axis=1)
    
    X = features_df.values
    print(f"   ✓ Features finais: {X.shape}")
    
    # 5. Validação temporal
    print("\n4. Aplicando validação temporal...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   ✓ Treino: {X_train.shape[0]} amostras")
    print(f"   ✓ Teste: {X_test.shape[0]} amostras")
    
    # 6. Normalização
    print("\n5. Aplicando normalização...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Seleção de features
    print("\n6. Selecionando features ótimas...")
    X_train_selected, feature_selector, selected_features = selecionar_features_otimas(
        X_train_scaled, y_train, k=60
    )
    X_test_selected = feature_selector.transform(X_test_scaled)
    
    # 8. Balanceamento
    print("\n7. Aplicando balanceamento...")
    X_train_balanced, y_train_balanced = aplicar_balanceamento_avancado(
        X_train_selected, y_train
    )
    
    # 9. Treinamento do ensemble
    print("\n8. Treinando ensemble super otimizado...")
    ensemble, modelos_individuais = criar_ensemble_super_otimizado(
        X_train_balanced, y_train_balanced
    )
    
    # 10. Avaliação
    print("\n9. Avaliando performance...")
    metricas = avaliar_modelo_completo(ensemble, X_test_selected, y_test)
    
    # 11. Resultados
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DA OTIMIZAÇÃO AVANÇADA:")
    print("=" * 60)
    print(f"• Acurácia: {formatar_porcentagem(metricas['accuracy'])}")
    print(f"• Precisão: {formatar_porcentagem(metricas['precision'])}")
    print(f"• Recall: {formatar_porcentagem(metricas['recall'])}")
    print(f"• F1-Score: {formatar_porcentagem(metricas['f1_score'])}")
    print(f"• ROC-AUC: {formatar_porcentagem(metricas['roc_auc'])}")
    
    # Status da meta
    if metricas['accuracy'] >= 0.85:
        status = "✅ META ATINGIDA"
    else:
        status = "❌ ABAIXO DA META"
    
    print(f"\n🎯 Status: {status}")
    
    # 12. Salvar modelo otimizado
    print("\n10. Salvando modelo otimizado...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar pipeline completo
    pipeline_completo = {
        'scaler': scaler,
        'feature_selector': feature_selector,
        'modelo': ensemble,
        'selected_features': selected_features,
        'metricas': metricas
    }
    
    # Salvar modelo
    modelo_path = f"modelos/modelo_super_otimizado_{timestamp}.pkl"
    os.makedirs("modelos", exist_ok=True)
    with open(modelo_path, 'wb') as f:
        pickle.dump(pipeline_completo, f)
    print(f"   ✓ Modelo salvo: {modelo_path}")
    
    # Salvar relatório
    relatorio = {
        'timestamp': timestamp,
        'total_features': X.shape[1],
        'features_selecionadas': len(selected_features),
        'amostras_treino': X_train.shape[0],
        'amostras_teste': X_test.shape[0],
        'metricas': {
            'accuracy': f"{metricas['accuracy']:.4f}",
            'precision': f"{metricas['precision']:.4f}",
            'recall': f"{metricas['recall']:.4f}",
            'f1_score': f"{metricas['f1_score']:.4f}",
            'roc_auc': f"{metricas['roc_auc']:.4f}"
        },
        'metricas_formatadas': {
            'accuracy': formatar_porcentagem(metricas['accuracy']),
            'precision': formatar_porcentagem(metricas['precision']),
            'recall': formatar_porcentagem(metricas['recall']),
            'f1_score': formatar_porcentagem(metricas['f1_score']),
            'roc_auc': formatar_porcentagem(metricas['roc_auc'])
        },
        'meta_atingida': metricas['accuracy'] >= 0.85,
        'modelos_ensemble': list(modelos_individuais.keys())
    }
    
    relatorio_path = f"experimentos/resultados/otimizacao_avancada_{timestamp}.json"
    os.makedirs("experimentos/resultados", exist_ok=True)
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Relatório salvo: {relatorio_path}")
    
    print("\n🎉 OTIMIZAÇÃO AVANÇADA FINALIZADA!")
    
    return metricas['accuracy'] >= 0.85

if __name__ == "__main__":
    sucesso = main()
    if sucesso:
        print("\n✅ META DE 85-90% ATINGIDA COM SUCESSO!")
    else:
        print("\n⚠️ META NÃO ATINGIDA - NECESSÁRIO AJUSTES ADICIONAIS")