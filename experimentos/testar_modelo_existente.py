import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix, 
    roc_auc_score
)
from sklearn.model_selection import train_test_split

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dados.dados import carregar_dados, preparar_dados
from experimentos.feature_engineering import FeatureEngineeringLotofacil

def formatar_porcentagem(valor):
    """Converte decimal para porcentagem formatada (ex: 0.85 -> 85.0%)"""
    return f"{valor * 100:.1f}%"

def testar_modelo_existente():
    print("=== TESTE DE MODELO EXISTENTE ===")
    print(f"Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Carregar dados validados
    print("\n1. Carregando dados validados...")
    try:
        df = carregar_dados()
        print(f"   ✓ Dados carregados: {len(df)} concursos")
        
        # Preparar dados básicos
        # Usar o DataFrame original que já tem a coluna 'Ganhou'
        df_preparado = df.copy()
        print(f"   ✓ Dados preparados: {len(df_preparado)} registros")
        print(f"   ✓ Período: {df_preparado['Data Sorteio'].min()} a {df_preparado['Data Sorteio'].max()}")
        print(f"   ✓ Concursos com ganhadores: {df_preparado['Ganhou'].sum()}/{len(df_preparado)} ({formatar_porcentagem(df_preparado['Ganhou'].mean())})")
        
    except Exception as e:
        print(f"   ✗ Erro ao carregar dados: {e}")
        return
    
    # 2. Carregar modelo ensemble
    print("\n2. Carregando modelo ensemble...")
    modelo_path = "modelo/optimized_models/ensemble_otimizado.pkl"
    
    try:
        with open(modelo_path, 'rb') as f:
            modelo = pickle.load(f)
        print(f"   ✓ Modelo carregado: {type(modelo).__name__}")
        print(f"   ✓ Estimadores: {[name for name, _ in modelo.estimators]}")
        
    except Exception as e:
        print(f"   ✗ Erro ao carregar modelo: {e}")
        return
    
    # 3. Preparar features (começando apenas com estatísticas)
    print("\n3. Preparando features...")
    try:
        # Inicializar feature engineer
        fe = FeatureEngineeringLotofacil()
        
        # Começar apenas com features estatísticas
        print("   ℹ️ Criando features estatísticas...")
        features_df = fe.criar_features_estatisticas(df_preparado)
        print(f"   ✓ Features estatísticas: {features_df.shape[1]} colunas")
        
        # Tentar adicionar features temporais de forma mais robusta
        print("   ℹ️ Tentando adicionar features temporais...")
        try:
            # Criar um subset menor para teste das features temporais
            df_subset = df_preparado.head(100)  # Usar apenas 100 concursos para teste
            features_temp = fe.criar_features_temporais(df_subset)
            
            if not features_temp.empty and len(features_temp.columns) > 1:
                print(f"   ✓ Features temporais: {features_temp.shape[1]} colunas")
                
                # Alinhar com features estatísticas
                if 'concurso' in features_temp.columns and 'concurso' in features_df.columns:
                    features_df = pd.merge(features_df, features_temp, on='concurso', how='left')
                    print(f"   ✓ Features combinadas: {features_df.shape[1]} colunas")
                else:
                    print("   ⚠️ Não foi possível alinhar features temporais (sem coluna concurso)")
            else:
                print("   ⚠️ Features temporais vazias ou inválidas")
        except Exception as temp_error:
            print(f"   ⚠️ Erro nas features temporais: {temp_error}")
            print("   ℹ️ Continuando apenas com features estatísticas")
        
        # Preparar targets alinhados com as features
        if 'concurso' in features_df.columns:
            print(f"   ℹ️ Alinhando targets usando coluna 'concurso'")
            # Alinhar targets com os concursos das features
            concursos_features = features_df['concurso'].values
            targets_alinhados = []
            concursos_nao_encontrados = 0
            
            for concurso in concursos_features:
                idx = df_preparado[df_preparado['Concurso'] == concurso].index
                if len(idx) > 0:
                    ganhou_valor = df_preparado.loc[idx[0], 'Ganhou']
                    # Garantir que seja binário (0 ou 1)
                    targets_alinhados.append(1 if ganhou_valor > 0 else 0)
                else:
                    targets_alinhados.append(0)  # Default para concursos não encontrados
                    concursos_nao_encontrados += 1
            
            y = np.array(targets_alinhados)
            print(f"   ℹ️ Concursos não encontrados: {concursos_nao_encontrados}")
            print(f"   ℹ️ Valores únicos nos targets alinhados: {np.unique(y)}")
            
            # Remover coluna concurso das features
            features_df = features_df.drop('concurso', axis=1)
        else:
            print(f"   ℹ️ Usando targets diretos (sem coluna concurso)")
            # Usar targets diretos se não há coluna concurso
            y_raw = df_preparado['Ganhou'].values[:len(features_df)]
            # Garantir que seja binário (0 ou 1)
            y = np.array([1 if val > 0 else 0 for val in y_raw])
            print(f"   ℹ️ Valores únicos nos targets diretos: {np.unique(y)}")
        
        # Converter features para array numpy
        X = features_df.values
        
        print(f"   ✓ Features extraídas: {X.shape}")
        print(f"   ✓ Targets: {len(y)} ({np.sum(y)} ganhadores)")
        print(f"   ✓ Features disponíveis: {list(features_df.columns)[:10]}...")  # Mostrar primeiras 10
        
        # Ajustar para 81 features esperadas pelo modelo
        if X.shape[1] < 81:
            print(f"   ⚠️ Preenchendo {81 - X.shape[1]} features faltantes com zeros")
            zeros_padding = np.zeros((X.shape[0], 81 - X.shape[1]))
            X = np.hstack([X, zeros_padding])
            print(f"   ✓ Features ajustadas para: {X.shape}")
        elif X.shape[1] > 81:
            print(f"   ⚠️ Reduzindo de {X.shape[1]} para 81 features (primeiras 81)")
            X = X[:, :81]
            print(f"   ✓ Features ajustadas para: {X.shape}")
        
    except Exception as e:
        print(f"   ✗ Erro ao preparar features: {e}")
        print(f"   Detalhes do erro: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Dividir dados temporalmente (80% treino, 20% teste)
    print("\n4. Dividindo dados temporalmente...")
    try:
        # Divisão temporal: primeiros 80% para treino, últimos 20% para teste
        split_idx = int(len(X) * 0.8)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"   ✓ Treino: {len(X_train)} amostras")
        print(f"   ✓ Teste: {len(X_test)} amostras")
        print(f"   ✓ Ganhadores no teste: {np.sum(y_test)}")
        
    except Exception as e:
        print(f"   ✗ Erro na divisão temporal: {e}")
        return
    
    # 5. Testar modelo
    print("\n5. Testando modelo...")
    try:
        # Fazer predições
        y_pred = modelo.predict(X_test)
        
        print(f"   ℹ️ Classes encontradas nas predições: {np.unique(y_pred)}")
        print(f"   ℹ️ Classes encontradas nos targets de teste: {np.unique(y_test)}")
        print(f"   ℹ️ Primeiras 10 predições: {y_pred[:10]}")
        print(f"   ℹ️ Primeiros 10 targets: {y_test[:10]}")
        
        # Tentar obter probabilidades se disponível
        try:
            if hasattr(modelo, 'predict_proba'):
                y_pred_proba = modelo.predict_proba(X_test)
                print(f"   ℹ️ Shape das probabilidades: {y_pred_proba.shape}")
            elif hasattr(modelo, 'decision_function'):
                y_pred_proba = modelo.decision_function(X_test)
            else:
                # Usar as predições como proxy para probabilidades
                y_pred_proba = y_pred.astype(float)
        except Exception as prob_error:
            print(f"   ⚠️ Não foi possível obter probabilidades: {prob_error}")
            y_pred_proba = y_pred.astype(float)
        
        # Verificar se é problema binário ou multiclass
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        print(f"   ℹ️ Classes encontradas: {unique_classes}")
        
        # Calcular métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        
        # Determinar se é binário ou multiclass
        if len(unique_classes) == 2:
            # Problema binário
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            # Problema multiclass - usar média weighted
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"   ✓ Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   ✓ Precisão: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   ✓ Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"   ✓ F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        
        # Tentar calcular ROC-AUC se temos probabilidades válidas
        try:
            if len(unique_classes) == 2 and len(np.unique(y_pred_proba)) > 1:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                print(f"   ✓ ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
            else:
                print(f"   ⚠️ ROC-AUC não calculável (multiclass ou probabilidades constantes)")
        except Exception as auc_error:
            print(f"   ⚠️ Erro ao calcular ROC-AUC: {auc_error}")
        
        # Relatório detalhado
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        print("\n   📊 MÉTRICAS DETALHADAS:")
        print(f"   • Acurácia Geral: {formatar_porcentagem(accuracy)}")
        print(f"   • Precisão (Não Ganhou): {formatar_porcentagem(report['0']['precision'])}")
        print(f"   • Recall (Não Ganhou): {formatar_porcentagem(report['0']['recall'])}")
        print(f"   • F1-Score (Não Ganhou): {formatar_porcentagem(report['0']['f1-score'])}")
        
        if '1' in report:
            print(f"   • Precisão (Ganhou): {formatar_porcentagem(report['1']['precision'])}")
            print(f"   • Recall (Ganhou): {formatar_porcentagem(report['1']['recall'])}")
            print(f"   • F1-Score (Ganhou): {formatar_porcentagem(report['1']['f1-score'])}")
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   📈 MATRIZ DE CONFUSÃO:")
        print(f"   ✓ Verdadeiros Negativos: {cm[0,0]}")
        print(f"   ✓ Falsos Positivos: {cm[0,1]}")
        print(f"   ✓ Falsos Negativos: {cm[1,0]}")
        print(f"   ✓ Verdadeiros Positivos: {cm[1,1]}")
        
        # Análise por classe
        print("\n   🎯 ANÁLISE POR CLASSE:")
        print(f"   ✓ Classe 0 (Sem ganhador): Precisão={report['0']['precision']*100:.2f}%, Recall={report['0']['recall']*100:.2f}%")
        print(f"   ✓ Classe 1 (Com ganhador): Precisão={report['1']['precision']*100:.2f}%, Recall={report['1']['recall']*100:.2f}%")
        
    except Exception as e:
        print(f"   ✗ Erro no teste do modelo: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Análise de probabilidades
    print("\n6. Analisando probabilidades...")
    prob_ganhou = None
    try:
        # Estatísticas das probabilidades
        if hasattr(y_pred_proba, 'shape') and len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            prob_ganhou = y_pred_proba[:, 1]
        elif hasattr(y_pred_proba, 'shape') and len(y_pred_proba.shape) > 1:
            prob_ganhou = y_pred_proba[:, 0]
        else:
            prob_ganhou = y_pred_proba
        
        print(f"   • Probabilidade média de ganhar: {formatar_porcentagem(np.mean(prob_ganhou))}")
        print(f"   • Probabilidade máxima: {formatar_porcentagem(np.max(prob_ganhou))}")
        print(f"   • Probabilidade mínima: {formatar_porcentagem(np.min(prob_ganhou))}")
        print(f"   • Desvio padrão: {formatar_porcentagem(np.std(prob_ganhou))}")
        
    except Exception as e:
        print(f"   ✗ Erro na análise de probabilidades: {e}")
        # Usar valores padrão se não conseguir calcular probabilidades
        prob_ganhou = y_pred.astype(float)
    
    # 7. Salvar relatório
    print("\n7. Salvando relatório...")
    relatorio = None
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Garantir que prob_ganhou existe
        if prob_ganhou is None:
            prob_ganhou = y_pred.astype(float)
        
        relatorio = {
            'timestamp': timestamp,
            'teste_realizado': 'Modelo Ensemble Existente',
            'dados': {
                'total_concursos': len(df),
                'total_features': X.shape[1],
                'treino_amostras': len(X_train),
                'teste_amostras': len(X_test),
                'ganhadores_teste': int(np.sum(y_test))
            },
            'modelo': {
                'tipo': type(modelo).__name__,
                'estimadores': [name for name, _ in modelo.estimators] if hasattr(modelo, 'estimators') else ['Modelo Único'],
                'path': modelo_path
            },
            'metricas': {
                'acuracia_geral': float(accuracy),
                'acuracia_formatada': formatar_porcentagem(accuracy),
                'relatorio_classificacao': report,
                'matriz_confusao': cm.tolist(),
                'probabilidades': {
                    'media': float(np.mean(prob_ganhou)),
                    'maxima': float(np.max(prob_ganhou)),
                    'minima': float(np.min(prob_ganhou)),
                    'desvio_padrao': float(np.std(prob_ganhou))
                }
            },
            'analise': {
                'status': 'TESTADO',
                'observacoes': [
                    f"Modelo testado com {len(X_test)} amostras de teste",
                    f"Acurácia atual: {formatar_porcentagem(accuracy)}",
                    "Meta desejada: 85-90% de acurácia",
                    "Necessário retreinamento para melhorar performance" if accuracy < 0.85 else "Performance dentro da meta"
                ]
            }
        }
        
        # Salvar relatório
        os.makedirs('experimentos/resultados', exist_ok=True)
        relatorio_path = f'experimentos/resultados/teste_modelo_existente_{timestamp}.json'
        
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Relatório salvo: {relatorio_path}")
        
    except Exception as e:
        print(f"   ✗ Erro ao salvar relatório: {e}")
        # Criar relatório básico em caso de erro
        relatorio = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'teste_realizado': 'Modelo Ensemble Existente',
            'status': 'ERRO_NO_SALVAMENTO',
            'acuracia': float(accuracy) if 'accuracy' in locals() else 0.0
        }
    
    # 8. Conclusão
    print("\n" + "="*50)
    print("📋 RESUMO DO TESTE:")
    print(f"• Modelo: Ensemble Otimizado")
    print(f"• Dados: {len(df)} concursos validados")
    print(f"• Acurácia Atual: {formatar_porcentagem(accuracy)}")
    print(f"• Meta Desejada: 85.0% - 90.0%")
    
    if accuracy >= 0.85:
        print(f"• Status: ✅ DENTRO DA META")
    else:
        print(f"• Status: ⚠️  ABAIXO DA META - NECESSÁRIO RETREINAMENTO")
    
    print("="*50)
    
    return relatorio

if __name__ == "__main__":
    testar_modelo_existente()