#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Final de Otimização Extrema para Modelo Lotofácil
Objetivo: Atingir 85-90% de acurácia com técnicas avançadas de ML
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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

def criar_features_extremas(df):
    """Cria features extremamente avançadas com múltiplas técnicas"""
    print("   🚀 Criando features extremamente avançadas...")
    
    fe = FeatureEngineeringLotofacil()
    
    # 1. Features estatísticas
    features_estat = fe.criar_features_estatisticas(df)
    print(f"      ✓ Features estatísticas: {features_estat.shape[1]} colunas")
    
    # 2. Features customizadas avançadas
    features_custom = criar_features_matematicas_avancadas(df)
    print(f"      ✓ Features matemáticas: {features_custom.shape[1]} colunas")
    
    # 3. Features de interação
    features_interacao = criar_features_interacao(df)
    print(f"      ✓ Features de interação: {features_interacao.shape[1]} colunas")
    
    # 4. Features de tendência histórica
    features_tendencia = criar_features_tendencia_historica(df)
    print(f"      ✓ Features de tendência: {features_tendencia.shape[1]} colunas")
    
    # Combinar todas as features
    features_list = [features_estat, features_custom, features_interacao, features_tendencia]
    features_completas = pd.concat(features_list, axis=1)
    
    # Remover colunas duplicadas e com variância zero
    features_completas = features_completas.loc[:, ~features_completas.columns.duplicated()]
    features_completas = features_completas.loc[:, features_completas.var() > 1e-6]
    
    print(f"   ✅ Total de features criadas: {features_completas.shape[1]}")
    return features_completas

def criar_features_matematicas_avancadas(df):
    """Cria features matemáticas super avançadas"""
    features_math = pd.DataFrame(index=df.index)
    
    numeros_cols = [f'Bola_{i:02d}' for i in range(1, 16)]
    
    for idx, row in df.iterrows():
        numeros = [row[col] for col in numeros_cols if col in row and pd.notna(row[col])]
        
        if len(numeros) == 15:
            numeros = np.array(numeros)
            
            # Features estatísticas avançadas
            features_math.loc[idx, 'skewness'] = pd.Series(numeros).skew()
            features_math.loc[idx, 'kurtosis'] = pd.Series(numeros).kurtosis()
            features_math.loc[idx, 'coef_variacao'] = np.std(numeros) / np.mean(numeros) if np.mean(numeros) > 0 else 0
            
            # Features geométricas
            features_math.loc[idx, 'media_geometrica'] = np.exp(np.mean(np.log(numeros)))
            features_math.loc[idx, 'media_harmonica'] = len(numeros) / np.sum(1.0/numeros)
            
            # Features de dispersão
            q1, q3 = np.percentile(numeros, [25, 75])
            features_math.loc[idx, 'iqr'] = q3 - q1
            features_math.loc[idx, 'mad'] = np.median(np.abs(numeros - np.median(numeros)))
            
            # Features de posição relativa
            features_math.loc[idx, 'pos_mediana'] = np.where(np.sort(numeros) == np.median(numeros))[0][0] if len(np.where(np.sort(numeros) == np.median(numeros))[0]) > 0 else 7
            
            # Features de densidade
            hist, _ = np.histogram(numeros, bins=5, range=(1, 25))
            features_math.loc[idx, 'densidade_max'] = np.max(hist)
            features_math.loc[idx, 'densidade_min'] = np.min(hist)
            features_math.loc[idx, 'densidade_var'] = np.var(hist)
            
            # Features de padrões numéricos
            diffs = np.diff(np.sort(numeros))
            features_math.loc[idx, 'diff_media'] = np.mean(diffs)
            features_math.loc[idx, 'diff_std'] = np.std(diffs)
            features_math.loc[idx, 'diff_max'] = np.max(diffs)
            features_math.loc[idx, 'diff_min'] = np.min(diffs)
            
            # Features de simetria
            centro = 13  # Centro da faixa 1-25
            distancias = np.abs(numeros - centro)
            features_math.loc[idx, 'simetria_centro'] = np.mean(distancias)
            
            # Features de concentração
            features_math.loc[idx, 'concentracao_baixa'] = np.sum(numeros <= 8)
            features_math.loc[idx, 'concentracao_media'] = np.sum((numeros > 8) & (numeros <= 17))
            features_math.loc[idx, 'concentracao_alta'] = np.sum(numeros > 17)
            
        else:
            # Valores padrão
            for col in ['skewness', 'kurtosis', 'coef_variacao', 'media_geometrica', 'media_harmonica',
                       'iqr', 'mad', 'pos_mediana', 'densidade_max', 'densidade_min', 'densidade_var',
                       'diff_media', 'diff_std', 'diff_max', 'diff_min', 'simetria_centro',
                       'concentracao_baixa', 'concentracao_media', 'concentracao_alta']:
                features_math.loc[idx, col] = 0
    
    return features_math.fillna(0)

def criar_features_interacao(df):
    """Cria features de interação entre variáveis"""
    features_int = pd.DataFrame(index=df.index)
    
    numeros_cols = [f'Bola_{i:02d}' for i in range(1, 16)]
    
    for idx, row in df.iterrows():
        numeros = [row[col] for col in numeros_cols if col in row and pd.notna(row[col])]
        
        if len(numeros) == 15:
            numeros = np.array(numeros)
            
            # Interações entre estatísticas básicas
            soma = np.sum(numeros)
            media = np.mean(numeros)
            std = np.std(numeros)
            
            features_int.loc[idx, 'soma_x_media'] = soma * media
            features_int.loc[idx, 'soma_div_std'] = soma / std if std > 0 else 0
            features_int.loc[idx, 'media_x_std'] = media * std
            
            # Interações de paridade
            pares = np.sum(numeros % 2 == 0)
            impares = np.sum(numeros % 2 == 1)
            features_int.loc[idx, 'razao_par_impar'] = pares / impares if impares > 0 else 0
            features_int.loc[idx, 'produto_par_impar'] = pares * impares
            
            # Interações de posição
            baixos = np.sum(numeros <= 12)
            altos = np.sum(numeros >= 14)
            features_int.loc[idx, 'razao_baixo_alto'] = baixos / altos if altos > 0 else 0
            features_int.loc[idx, 'produto_baixo_alto'] = baixos * altos
            
            # Interações de quadrantes
            q1 = np.sum((numeros >= 1) & (numeros <= 6))
            q2 = np.sum((numeros >= 7) & (numeros <= 12))
            q3 = np.sum((numeros >= 13) & (numeros <= 18))
            q4 = np.sum((numeros >= 19) & (numeros <= 25))
            
            features_int.loc[idx, 'q1_x_q4'] = q1 * q4
            features_int.loc[idx, 'q2_x_q3'] = q2 * q3
            features_int.loc[idx, 'diagonal_principal'] = q1 + q4
            features_int.loc[idx, 'diagonal_secundaria'] = q2 + q3
            
        else:
            # Valores padrão
            for col in ['soma_x_media', 'soma_div_std', 'media_x_std', 'razao_par_impar', 'produto_par_impar',
                       'razao_baixo_alto', 'produto_baixo_alto', 'q1_x_q4', 'q2_x_q3', 'diagonal_principal', 'diagonal_secundaria']:
                features_int.loc[idx, col] = 0
    
    return features_int.fillna(0)

def criar_features_tendencia_historica(df):
    """Cria features baseadas em tendências históricas"""
    features_tend = pd.DataFrame(index=df.index)
    
    numeros_cols = [f'Bola_{i:02d}' for i in range(1, 16)]
    
    # Calcular frequências históricas
    freq_historica = {i: 0 for i in range(1, 26)}
    
    for idx, row in df.iterrows():
        numeros = [row[col] for col in numeros_cols if col in row and pd.notna(row[col])]
        
        if len(numeros) == 15:
            # Atualizar frequências até este ponto
            for num in numeros:
                freq_historica[num] += 1
            
            # Features baseadas em frequência histórica
            freqs_atuais = [freq_historica[num] for num in numeros]
            features_tend.loc[idx, 'freq_media_historica'] = np.mean(freqs_atuais)
            features_tend.loc[idx, 'freq_std_historica'] = np.std(freqs_atuais)
            features_tend.loc[idx, 'freq_max_historica'] = np.max(freqs_atuais)
            features_tend.loc[idx, 'freq_min_historica'] = np.min(freqs_atuais)
            
            # Features de "calor" dos números
            total_jogos = idx + 1
            freq_relativas = [freq_historica[num] / total_jogos for num in numeros]
            features_tend.loc[idx, 'calor_medio'] = np.mean(freq_relativas)
            features_tend.loc[idx, 'calor_max'] = np.max(freq_relativas)
            features_tend.loc[idx, 'calor_min'] = np.min(freq_relativas)
            
            # Features de desvio da frequência esperada
            freq_esperada = total_jogos * 15 / 25  # Frequência esperada para cada número
            desvios = [abs(freq_historica[num] - freq_esperada) for num in numeros]
            features_tend.loc[idx, 'desvio_medio_freq'] = np.mean(desvios)
            features_tend.loc[idx, 'desvio_max_freq'] = np.max(desvios)
            
        else:
            # Valores padrão
            for col in ['freq_media_historica', 'freq_std_historica', 'freq_max_historica', 'freq_min_historica',
                       'calor_medio', 'calor_max', 'calor_min', 'desvio_medio_freq', 'desvio_max_freq']:
                features_tend.loc[idx, col] = 0
    
    return features_tend.fillna(0)

def criar_ensemble_extremo(X_train, y_train):
    """Cria ensemble extremamente otimizado com grid search"""
    print("   🎯 Criando ensemble extremo com grid search...")
    
    # Definir modelos com parâmetros para otimização
    modelos_params = {
        'rf': {
            'modelo': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gb': {
            'modelo': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        },
        'et': {
            'modelo': ExtraTreesClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
        }
    }
    
    # Otimizar cada modelo
    modelos_otimizados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for nome, config in modelos_params.items():
        print(f"      🔄 Otimizando {nome.upper()}...")
        
        grid_search = GridSearchCV(
            config['modelo'],
            config['params'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        modelos_otimizados[nome] = grid_search.best_estimator_
        print(f"         ✓ Melhor score: {grid_search.best_score_:.4f}")
    
    # Criar ensemble final
    ensemble = VotingClassifier(
        estimators=[(nome, modelo) for nome, modelo in modelos_otimizados.items()],
        voting='soft'
    )
    
    print("      🔄 Treinando ensemble final...")
    ensemble.fit(X_train, y_train)
    
    return ensemble, modelos_otimizados

def aplicar_balanceamento_extremo(X_train, y_train):
    """Aplica balanceamento extremo com múltiplas técnicas"""
    print("   ⚖️ Aplicando balanceamento extremo...")
    
    # Verificar distribuição original
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"      📊 Distribuição original: {dict(zip(unique, counts))}")
    
    # Usar SMOTETomek que combina oversampling e undersampling
    smote_tomek = SMOTETomek(random_state=42)
    X_balanced, y_balanced = smote_tomek.fit_resample(X_train, y_train)
    
    # Verificar nova distribuição
    unique, counts = np.unique(y_balanced, return_counts=True)
    print(f"      📊 Distribuição balanceada: {dict(zip(unique, counts))}")
    
    return X_balanced, y_balanced

def selecionar_features_extremas(X_train, y_train, k=80):
    """Seleção extrema de features com múltiplos métodos"""
    print(f"   🎯 Seleção extrema de {k} melhores features...")
    
    # Método 1: SelectKBest
    selector_kbest = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    X_kbest = selector_kbest.fit_transform(X_train, y_train)
    
    # Método 2: RFE com Random Forest
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(rf_selector, n_features_to_select=min(k, X_train.shape[1]))
    X_rfe = rfe.fit_transform(X_train, y_train)
    
    # Combinar seleções (interseção das features selecionadas)
    features_kbest = set(selector_kbest.get_support(indices=True))
    features_rfe = set(rfe.get_support(indices=True))
    features_intersecao = features_kbest.intersection(features_rfe)
    
    if len(features_intersecao) < 30:  # Garantir mínimo de features
        features_finais = list(features_kbest.union(features_rfe))[:k]
    else:
        features_finais = list(features_intersecao)
    
    # Aplicar seleção final
    X_selected = X_train[:, features_finais]
    
    print(f"      ✓ Features selecionadas: {len(features_finais)}")
    
    return X_selected, features_finais

def avaliar_modelo_extremo(modelo, X_test, y_test, features_selecionadas):
    """Avaliação extrema do modelo"""
    print("   📊 Avaliação extrema do modelo...")
    
    # Aplicar seleção de features no teste
    X_test_selected = X_test[:, features_selecionadas]
    
    # Predições
    y_pred = modelo.predict(X_test_selected)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC
    try:
        if hasattr(modelo, 'predict_proba'):
            y_proba = modelo.predict_proba(X_test_selected)
            if y_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = 0.5
    except:
        roc_auc = 0.5
    
    # Relatório detalhado
    print("\n" + "="*50)
    print("📊 RELATÓRIO DETALHADO:")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def main():
    """Função principal de otimização extrema"""
    print("🚀 INICIANDO OTIMIZAÇÃO EXTREMA DO MODELO LOTOFÁCIL")
    print("=" * 70)
    
    # 1. Carregar dados
    print("\n1. Carregando dados...")
    df_dados = carregar_dados()
    print(f"   ✓ Dados carregados: {len(df_dados)} concursos")
    
    # 2. Criar features extremas
    print("\n2. Criando features extremas...")
    features_df = criar_features_extremas(df_dados)
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
    
    # 4. Preparar features finais
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
    
    # 6. Normalização robusta
    print("\n5. Aplicando normalização robusta...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Seleção extrema de features
    print("\n6. Seleção extrema de features...")
    X_train_selected, features_selecionadas = selecionar_features_extremas(
        X_train_scaled, y_train, k=min(80, X_train_scaled.shape[1])
    )
    
    # 8. Balanceamento extremo
    print("\n7. Balanceamento extremo...")
    X_train_balanced, y_train_balanced = aplicar_balanceamento_extremo(
        X_train_selected, y_train
    )
    
    # 9. Treinamento do ensemble extremo
    print("\n8. Treinamento do ensemble extremo...")
    ensemble, modelos_individuais = criar_ensemble_extremo(
        X_train_balanced, y_train_balanced
    )
    
    # 10. Avaliação extrema
    print("\n9. Avaliação extrema...")
    metricas = avaliar_modelo_extremo(ensemble, X_test_scaled, y_test, features_selecionadas)
    
    # 11. Resultados finais
    print("\n" + "=" * 70)
    print("🏆 RESULTADOS DA OTIMIZAÇÃO EXTREMA:")
    print("=" * 70)
    print(f"• Acurácia: {formatar_porcentagem(metricas['accuracy'])}")
    print(f"• Precisão: {formatar_porcentagem(metricas['precision'])}")
    print(f"• Recall: {formatar_porcentagem(metricas['recall'])}")
    print(f"• F1-Score: {formatar_porcentagem(metricas['f1_score'])}")
    print(f"• ROC-AUC: {formatar_porcentagem(metricas['roc_auc'])}")
    
    # Status da meta
    if metricas['accuracy'] >= 0.85:
        status = "✅ META ATINGIDA - EXCELENTE!"
        cor = "🟢"
    elif metricas['accuracy'] >= 0.80:
        status = "🟡 PRÓXIMO DA META - BOM RESULTADO"
        cor = "🟡"
    else:
        status = "❌ ABAIXO DA META - NECESSÁRIO MAIS AJUSTES"
        cor = "🔴"
    
    print(f"\n{cor} Status: {status}")
    
    # 12. Salvar modelo final
    print("\n10. Salvando modelo final extremo...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Pipeline completo
    pipeline_final = {
        'scaler': scaler,
        'features_selecionadas': features_selecionadas,
        'modelo': ensemble,
        'modelos_individuais': modelos_individuais,
        'metricas': metricas,
        'timestamp': timestamp
    }
    
    # Salvar modelo
    modelo_path = f"modelos/modelo_final_extremo_{timestamp}.pkl"
    os.makedirs("modelos", exist_ok=True)
    with open(modelo_path, 'wb') as f:
        pickle.dump(pipeline_final, f)
    print(f"   ✓ Modelo salvo: {modelo_path}")
    
    # Relatório final
    relatorio_final = {
        'timestamp': timestamp,
        'versao': 'EXTREMA',
        'total_features_originais': X.shape[1],
        'features_selecionadas': len(features_selecionadas),
        'amostras_treino': X_train.shape[0],
        'amostras_teste': X_test.shape[0],
        'metricas_raw': {
            'accuracy': float(metricas['accuracy']),
            'precision': float(metricas['precision']),
            'recall': float(metricas['recall']),
            'f1_score': float(metricas['f1_score']),
            'roc_auc': float(metricas['roc_auc'])
        },
        'metricas_formatadas': {
            'accuracy': formatar_porcentagem(metricas['accuracy']),
            'precision': formatar_porcentagem(metricas['precision']),
            'recall': formatar_porcentagem(metricas['recall']),
            'f1_score': formatar_porcentagem(metricas['f1_score']),
            'roc_auc': formatar_porcentagem(metricas['roc_auc'])
        },
        'meta_85_90_atingida': metricas['accuracy'] >= 0.85,
        'meta_80_atingida': metricas['accuracy'] >= 0.80,
        'modelos_ensemble': list(modelos_individuais.keys()),
        'tecnicas_aplicadas': [
            'Feature Engineering Extrema',
            'Seleção de Features Múltipla (KBest + RFE)',
            'Balanceamento SMOTETomek',
            'Normalização Robusta',
            'Grid Search Otimização',
            'Ensemble Voting Classifier',
            'Validação Temporal'
        ]
    }
    
    relatorio_path = f"experimentos/resultados/modelo_final_extremo_{timestamp}.json"
    os.makedirs("experimentos/resultados", exist_ok=True)
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        json.dump(relatorio_final, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Relatório salvo: {relatorio_path}")
    
    print("\n🎉 OTIMIZAÇÃO EXTREMA FINALIZADA!")
    
    return metricas['accuracy'] >= 0.85, metricas['accuracy']

if __name__ == "__main__":
    sucesso, acuracia_final = main()
    
    print("\n" + "="*70)
    if sucesso:
        print("🏆 PARABÉNS! META DE 85-90% ATINGIDA COM SUCESSO!")
        print(f"🎯 Acurácia Final: {acuracia_final*100:.2f}%")
    else:
        print(f"📊 Resultado Final: {acuracia_final*100:.2f}%")
        if acuracia_final >= 0.80:
            print("🟡 Resultado muito bom, próximo da meta!")
        else:
            print("🔴 Necessário mais ajustes para atingir a meta.")
    print("="*70)