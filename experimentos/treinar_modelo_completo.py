#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treinamento Completo de Modelo de Lotofácil
Objetivo: Alcançar acurácia de 85-90% com ensemble otimizado
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json
from pathlib import Path

# Imports para ML
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Imports locais
from dados.dados import carregar_dados
from feature_engineering import FeatureEngineeringLotofacil

def formatar_porcentagem(valor):
    """Converte decimal para porcentagem formatada"""
    return f"{valor * 100:.2f}%"

def criar_features_avancadas(df):
    """Cria features avançadas para melhorar a performance"""
    print("   🔧 Criando features avançadas...")
    
    # Inicializar feature engineering
    fe = FeatureEngineeringLotofacil()
    
    # Features estatísticas básicas
    features_estat = fe.criar_features_estatisticas(df)
    print(f"      ✓ Features estatísticas: {features_estat.shape[1]} colunas")
    
    # Features temporais
    features_temp = fe.criar_features_temporais(df)
    print(f"      ✓ Features temporais: {features_temp.shape[1]} colunas")
    
    # Combinar todas as features
    features_completas = pd.concat([features_estat, features_temp], axis=1)
    
    # Remover colunas duplicadas
    features_completas = features_completas.loc[:, ~features_completas.columns.duplicated()]
    
    print(f"   ✅ Total de features criadas: {len(features_completas.columns)}")
    return features_completas

def adicionar_features_padroes(features_df, df_original):
    """Adiciona features de padrões avançados"""
    print("   📊 Adicionando features de padrões...")
    
    # Converter colunas de números para análise
    numeros_cols = [f'Numero_{i}' for i in range(1, 16) if f'Numero_{i}' in df_original.columns]
    
    if numeros_cols:
        # Padrões de sequência
        features_df['sequencias_consecutivas'] = df_original[numeros_cols].apply(
            lambda row: contar_sequencias_consecutivas(sorted(row.values)), axis=1
        )
        
        # Padrões de distância
        features_df['distancia_media'] = df_original[numeros_cols].apply(
            lambda row: calcular_distancia_media(sorted(row.values)), axis=1
        )
        
        # Padrões de distribuição por dezenas
        for dezena in range(1, 3):  # 1-10, 11-20, 21-25
            inicio = (dezena - 1) * 10 + 1
            fim = min(dezena * 10, 25)
            col_name = f'dezena_{inicio}_{fim}'
            features_df[col_name] = df_original[numeros_cols].apply(
                lambda row: sum(1 for x in row.values if inicio <= x <= fim), axis=1
            )
    
    return features_df

def contar_sequencias_consecutivas(numeros):
    """Conta sequências consecutivas nos números"""
    if len(numeros) < 2:
        return 0
    
    sequencias = 0
    for i in range(len(numeros) - 1):
        if numeros[i+1] - numeros[i] == 1:
            sequencias += 1
    return sequencias

def calcular_distancia_media(numeros):
    """Calcula distância média entre números"""
    if len(numeros) < 2:
        return 0
    
    distancias = [numeros[i+1] - numeros[i] for i in range(len(numeros) - 1)]
    return np.mean(distancias)

def criar_ensemble_otimizado():
    """Cria ensemble otimizado com múltiplos algoritmos"""
    print("   🤖 Criando ensemble otimizado...")
    
    # Modelos base com hiperparâmetros otimizados
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    
    # Pipeline com normalização para SVM
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42, kernel='rbf'))
    ])
    
    # Pipeline com normalização para Logistic Regression
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Ensemble com voting
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('svm', svm_pipeline),
            ('lr', lr_pipeline)
        ],
        voting='soft'  # Usa probabilidades
    )
    
    return ensemble

def validacao_temporal(X, y, modelo, test_size=0.2):
    """Validação temporal (últimos dados como teste)"""
    print(f"   ⏰ Aplicando validação temporal ({int((1-test_size)*100)}% treino, {int(test_size*100)}% teste)...")
    
    # Divisão temporal (últimos dados como teste)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"   ✓ Treino: {len(X_train)} amostras")
    print(f"   ✓ Teste: {len(X_test)} amostras")
    print(f"   ✓ Distribuição treino - Classe 0: {sum(y_train == 0)}, Classe 1: {sum(y_train == 1)}")
    print(f"   ✓ Distribuição teste - Classe 0: {sum(y_test == 0)}, Classe 1: {sum(y_test == 1)}")
    
    return X_train, X_test, y_train, y_test

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo com métricas completas"""
    print("   📊 Avaliando modelo...")
    
    # Predições
    y_pred = modelo.predict(X_test)
    
    # Métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC se possível
    try:
        if hasattr(modelo, 'predict_proba'):
            y_prob = modelo.predict_proba(X_test)
            if len(np.unique(y_test)) == 2 and y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                roc_auc = None
        else:
            roc_auc = None
    except:
        roc_auc = None
    
    metricas = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return metricas, y_pred

def treinar_modelo_completo():
    """Função principal de treinamento"""
    print("🚀 INICIANDO TREINAMENTO COMPLETO DO MODELO")
    print("=" * 50)
    
    try:
        # 1. Carregar dados
        print("\n1. Carregando dados...")
        df = carregar_dados()
        print(f"   ✓ Dados carregados: {len(df)} concursos")
        
        # 2. Criar features avançadas
        print("\n2. Criando features avançadas...")
        features_df = criar_features_avancadas(df)
        print(f"   ✓ Features criadas: {features_df.shape}")
        
        # 3. Preparar targets
        print("\n3. Preparando targets...")
        y = df['Ganhou'].values
        print(f"   ✓ Targets preparados: {len(y)} amostras")
        print(f"   ✓ Distribuição - Classe 0: {sum(y == 0)}, Classe 1: {sum(y == 1)}")
        
        # 4. Remover coluna concurso se existir
        if 'concurso' in features_df.columns:
            features_df = features_df.drop('concurso', axis=1)
        
        X = features_df.values
        print(f"   ✓ Features finais: {X.shape}")
        
        # 5. Validação temporal
        print("\n4. Aplicando validação temporal...")
        X_train, X_test, y_train, y_test = validacao_temporal(X, y, None)
        
        # 6. Criar e treinar ensemble
        print("\n5. Treinando ensemble otimizado...")
        modelo = criar_ensemble_otimizado()
        
        print("   🔄 Iniciando treinamento...")
        modelo.fit(X_train, y_train)
        print("   ✅ Treinamento concluído!")
        
        # 7. Avaliar modelo
        print("\n6. Avaliando performance...")
        metricas, y_pred = avaliar_modelo(modelo, X_test, y_test)
        
        # 8. Exibir resultados
        print("\n" + "=" * 50)
        print("📊 RESULTADOS DO TREINAMENTO:")
        print("=" * 50)
        print(f"• Acurácia: {formatar_porcentagem(metricas['accuracy'])}")
        print(f"• Precisão: {formatar_porcentagem(metricas['precision'])}")
        print(f"• Recall: {formatar_porcentagem(metricas['recall'])}")
        print(f"• F1-Score: {formatar_porcentagem(metricas['f1_score'])}")
        if metricas['roc_auc']:
            print(f"• ROC-AUC: {formatar_porcentagem(metricas['roc_auc'])}")
        
        # 9. Verificar meta
        accuracy_pct = metricas['accuracy'] * 100
        if 85 <= accuracy_pct <= 90:
            status = "✅ DENTRO DA META (85-90%)"
        elif accuracy_pct > 90:
            status = "⚠️ ACIMA DA META - Verificar overfitting"
        else:
            status = "❌ ABAIXO DA META - Necessário ajustes"
        
        print(f"\n🎯 Status: {status}")
        
        # 10. Salvar modelo
        print("\n7. Salvando modelo...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar diretórios se não existirem
        Path("modelos").mkdir(exist_ok=True)
        Path("experimentos/resultados").mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo
        modelo_path = f"modelos/ensemble_otimizado_{timestamp}.pkl"
        joblib.dump(modelo, modelo_path)
        print(f"   ✓ Modelo salvo: {modelo_path}")
        
        # Salvar relatório
        relatorio = {
            'timestamp': timestamp,
            'modelo': 'Ensemble Otimizado (RF + GB + SVM + LR)',
            'dados': {
                'total_concursos': len(df),
                'features_shape': list(X.shape),
                'treino_amostras': len(X_train),
                'teste_amostras': len(X_test)
            },
            'metricas': {
                'accuracy': float(metricas['accuracy']),
                'precision': float(metricas['precision']),
                'recall': float(metricas['recall']),
                'f1_score': float(metricas['f1_score']),
                'roc_auc': float(metricas['roc_auc']) if metricas['roc_auc'] else None
            },
            'meta_85_90': 85 <= accuracy_pct <= 90,
            'status': status
        }
        
        relatorio_path = f"experimentos/resultados/treinamento_completo_{timestamp}.json"
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
        print(f"   ✓ Relatório salvo: {relatorio_path}")
        
        print("\n🎉 TREINAMENTO COMPLETO FINALIZADO!")
        return modelo, metricas
        
    except Exception as e:
        print(f"\n❌ ERRO NO TREINAMENTO: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    modelo, metricas = treinar_modelo_completo()