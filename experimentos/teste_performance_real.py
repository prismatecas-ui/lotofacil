import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestePerformanceReal:
    """
    Classe para teste de performance real do sistema de IA da Lotofácil
    """
    
    def __init__(self):
        self.modelo_path = "./modelo/saved_models/lotofacil_completo_20250919_081854"
        self.resultados = {
            'timestamp': datetime.now().isoformat(),
            'modelo_atual': {},
            'problemas_identificados': [],
            'melhorias_implementadas': [],
            'performance_antes': {},
            'performance_depois': {},
            'modelos_otimizados': {}
        }
        
    def carregar_dados_reais(self):
        """
        Carrega dados reais para teste
        """
        print("Carregando dados reais...")
        
        # Tentar diferentes fontes de dados
        dados = None
        
        # 1. Tentar base Excel
        if os.path.exists("./base/base_dados.xlsx"):
            try:
                dados = pd.read_excel("./base/base_dados.xlsx")
                print(f"Dados carregados do Excel: {len(dados)} registros")
            except Exception as e:
                print(f"Erro ao carregar Excel: {e}")
        
        # 2. Tentar cache JSON
        if dados is None and os.path.exists("./base/cache_concursos.json"):
            try:
                with open("./base/cache_concursos.json", 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                dados = pd.DataFrame(cache_data)
                print(f"Dados carregados do cache: {len(dados)} registros")
            except Exception as e:
                print(f"Erro ao carregar cache: {e}")
        
        # 3. Gerar dados sintéticos se necessário
        if dados is None:
            print("Gerando dados sintéticos para teste...")
            dados = self.gerar_dados_sinteticos()
        
        return dados
    
    def gerar_dados_sinteticos(self, n_concursos=1000):
        """
        Gera dados sintéticos baseados em padrões reais da Lotofácil
        """
        np.random.seed(42)
        dados = []
        
        for i in range(n_concursos):
            # Gerar 15 números únicos entre 1 e 25
            numeros = sorted(np.random.choice(range(1, 26), 15, replace=False))
            
            concurso = {'Concurso': i + 1}
            for j, num in enumerate(numeros, 1):
                concurso[f'Bola{j:02d}'] = num
            
            dados.append(concurso)
        
        return pd.DataFrame(dados)
    
    def preparar_dados_para_teste(self, dados):
        """
        Prepara dados para teste de performance
        """
        print("Preparando dados para teste...")
        
        # Identificar colunas de números
        colunas_numeros = []
        for i in range(1, 16):
            col_name = f'Bola{i:02d}'
            if col_name in dados.columns:
                colunas_numeros.append(col_name)
        
        if len(colunas_numeros) < 15:
            # Usar colunas numéricas disponíveis
            colunas_numericas = dados.select_dtypes(include=[np.number]).columns
            colunas_numeros = colunas_numericas[:15].tolist()
        
        # Criar features e targets
        X = []
        y = []
        
        for i in range(len(dados) - 1):
            # Usar sorteio atual para predizer próximo
            sorteio_atual = dados.iloc[i][colunas_numeros]
            proximo_sorteio = dados.iloc[i + 1][colunas_numeros]
            
            # Converter para vetor binário (1-25)
            vetor_atual = [0] * 25
            for num in sorteio_atual:
                if pd.notna(num) and 1 <= int(num) <= 25:
                    vetor_atual[int(num) - 1] = 1
            
            vetor_proximo = [0] * 25
            for num in proximo_sorteio:
                if pd.notna(num) and 1 <= int(num) <= 25:
                    vetor_proximo[int(num) - 1] = 1
            
            X.append(vetor_atual)
            y.append(vetor_proximo)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Dados preparados: X={X.shape}, y={y.shape}")
        return X, y
    
    def testar_modelo_atual(self, X, y):
        """
        Testa o modelo atual salvo
        """
        print("\n=== TESTANDO MODELO ATUAL ===")
        
        try:
            # Carregar modelo
            modelo_path = os.path.join(self.modelo_path, "modelo.h5")
            if os.path.exists(modelo_path):
                modelo = load_model(modelo_path)
                print(f"Modelo carregado: {modelo_path}")
                
                # Dividir dados para teste
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                # Fazer predições
                y_pred = modelo.predict(X_test)
                y_pred_binary = (y_pred > 0.5).astype(int)
                
                # Calcular métricas reais
                accuracy = accuracy_score(y_test.flatten(), y_pred_binary.flatten())
                precision = precision_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
                recall = recall_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
                f1 = f1_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
                
                # Calcular taxa de acerto por jogo (métrica mais realista)
                acertos_por_jogo = []
                for i in range(len(y_test)):
                    numeros_reais = np.where(y_test[i] == 1)[0] + 1
                    numeros_preditos = np.where(y_pred_binary[i] == 1)[0] + 1
                    acertos = len(set(numeros_reais) & set(numeros_preditos))
                    acertos_por_jogo.append(acertos)
                
                taxa_acerto_media = np.mean(acertos_por_jogo)
                taxa_acerto_percentual = (taxa_acerto_media / 15) * 100
                
                metricas_atuais = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'taxa_acerto_media': taxa_acerto_media,
                    'taxa_acerto_percentual': taxa_acerto_percentual,
                    'distribuicao_acertos': np.bincount(acertos_por_jogo, minlength=16).tolist()
                }
                
                self.resultados['performance_antes'] = metricas_atuais
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                print(f"Taxa de acerto média: {taxa_acerto_media:.2f} números por jogo")
                print(f"Taxa de acerto percentual: {taxa_acerto_percentual:.2f}%")
                
                return metricas_atuais, X_train, X_test, y_train, y_test
                
            else:
                print(f"Modelo não encontrado: {modelo_path}")
                self.resultados['problemas_identificados'].append("Modelo salvo não encontrado")
                return None, None, None, None, None
                
        except Exception as e:
            print(f"Erro ao testar modelo atual: {e}")
            self.resultados['problemas_identificados'].append(f"Erro ao carregar modelo: {str(e)}")
            return None, None, None, None, None
    
    def identificar_problemas_arquitetura(self, metricas_atuais):
        """
        Identifica problemas na arquitetura atual
        """
        print("\n=== IDENTIFICANDO PROBLEMAS ===")
        
        problemas = []
        
        if metricas_atuais:
            if metricas_atuais['taxa_acerto_percentual'] < 80:
                problemas.append("Taxa de acerto baixa (< 80%)")
            
            if metricas_atuais['accuracy'] < 0.8:
                problemas.append("Accuracy baixa (< 0.8)")
            
            if metricas_atuais['precision'] < 0.7:
                problemas.append("Precision baixa (< 0.7)")
            
            if metricas_atuais['recall'] < 0.7:
                problemas.append("Recall baixo (< 0.7)")
        
        # Problemas arquiteturais conhecidos
        problemas.extend([
            "Modelo simples sem ensemble",
            "Falta de feature engineering avançada",
            "Ausência de validação cruzada temporal",
            "Sem otimização de hiperparâmetros",
            "Arquitetura neural básica"
        ])
        
        self.resultados['problemas_identificados'] = problemas
        
        for problema in problemas:
            print(f"- {problema}")
        
        return problemas
    
    def implementar_feature_engineering_avancada(self, X, y):
        """
        Implementa feature engineering avançada
        """
        print("\n=== IMPLEMENTANDO FEATURE ENGINEERING AVANÇADA ===")
        
        X_melhorado = []
        
        for i, sample in enumerate(X):
            features_originais = sample.tolist()
            
            # Features estatísticas
            numeros_sorteados = np.where(np.array(sample) == 1)[0] + 1
            
            # 1. Soma dos números
            soma_numeros = np.sum(numeros_sorteados)
            
            # 2. Paridade (pares vs ímpares)
            pares = np.sum(numeros_sorteados % 2 == 0)
            impares = 15 - pares
            
            # 3. Distribuição por dezenas
            dezena_1 = np.sum((numeros_sorteados >= 1) & (numeros_sorteados <= 5))
            dezena_2 = np.sum((numeros_sorteados >= 6) & (numeros_sorteados <= 10))
            dezena_3 = np.sum((numeros_sorteados >= 11) & (numeros_sorteados <= 15))
            dezena_4 = np.sum((numeros_sorteados >= 16) & (numeros_sorteados <= 20))
            dezena_5 = np.sum((numeros_sorteados >= 21) & (numeros_sorteados <= 25))
            
            # 4. Sequências consecutivas
            consecutivos = 0
            for j in range(len(numeros_sorteados) - 1):
                if numeros_sorteados[j+1] - numeros_sorteados[j] == 1:
                    consecutivos += 1
            
            # 5. Distância média entre números
            if len(numeros_sorteados) > 1:
                distancias = np.diff(numeros_sorteados)
                distancia_media = np.mean(distancias)
                distancia_std = np.std(distancias)
            else:
                distancia_media = 0
                distancia_std = 0
            
            # 6. Features históricas (se disponível)
            if i > 0:
                # Repetições do sorteio anterior
                anterior = X[i-1]
                repeticoes = np.sum(sample * anterior)
            else:
                repeticoes = 0
            
            # Combinar todas as features
            features_avancadas = features_originais + [
                soma_numeros / 325,  # Normalizado
                pares / 15,
                impares / 15,
                dezena_1 / 15,
                dezena_2 / 15,
                dezena_3 / 15,
                dezena_4 / 15,
                dezena_5 / 15,
                consecutivos / 14,
                distancia_media / 25,
                distancia_std / 25,
                repeticoes / 15
            ]
            
            X_melhorado.append(features_avancadas)
        
        X_melhorado = np.array(X_melhorado)
        print(f"Features expandidas: {X.shape} -> {X_melhorado.shape}")
        
        self.resultados['melhorias_implementadas'].append("Feature engineering avançada")
        
        return X_melhorado
    
    def criar_modelo_ensemble_otimizado(self, input_shape):
        """
        Cria modelo ensemble otimizado
        """
        print("\n=== CRIANDO MODELO ENSEMBLE OTIMIZADO ===")
        
        # Modelo Neural Network melhorado
        def criar_nn_otimizada():
            model = tf.keras.Sequential([
                layers.Input(shape=(input_shape,)),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(25, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            return model
        
        # XGBoost otimizado
        def criar_xgboost_otimizado():
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        # Random Forest otimizado
        def criar_rf_otimizado():
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        
        modelos = {
            'neural_network': criar_nn_otimizada(),
            'xgboost': criar_xgboost_otimizado(),
            'random_forest': criar_rf_otimizado()
        }
        
        self.resultados['melhorias_implementadas'].append("Ensemble de modelos otimizados")
        
        return modelos
    
    def treinar_com_validacao_temporal(self, X, y, modelos):
        """
        Treina modelos com validação cruzada temporal
        """
        print("\n=== TREINAMENTO COM VALIDAÇÃO TEMPORAL ===")
        
        # Usar TimeSeriesSplit para validação temporal
        tscv = TimeSeriesSplit(n_splits=5)
        
        resultados_modelos = {}
        
        for nome, modelo in modelos.items():
            print(f"\nTreinando {nome}...")
            
            scores_accuracy = []
            scores_precision = []
            scores_recall = []
            scores_f1 = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                if nome == 'neural_network':
                    # Treinar NN
                    callbacks = [
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=5)
                    ]
                    
                    modelo.fit(
                        X_train_fold, y_train_fold,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val_fold, y_val_fold),
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    y_pred = modelo.predict(X_val_fold)
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    
                else:
                    # Treinar modelos sklearn
                    # Para modelos multi-output, treinar para cada posição
                    y_pred_binary = np.zeros_like(y_val_fold)
                    
                    for pos in range(25):
                        modelo_pos = type(modelo)(**modelo.get_params())
                        modelo_pos.fit(X_train_fold, y_train_fold[:, pos])
                        y_pred_binary[:, pos] = modelo_pos.predict(X_val_fold)
                
                # Calcular métricas
                accuracy = accuracy_score(y_val_fold.flatten(), y_pred_binary.flatten())
                precision = precision_score(y_val_fold.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
                recall = recall_score(y_val_fold.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
                f1 = f1_score(y_val_fold.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
                
                scores_accuracy.append(accuracy)
                scores_precision.append(precision)
                scores_recall.append(recall)
                scores_f1.append(f1)
                
                print(f"  Fold {fold+1}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
            
            # Métricas médias
            resultados_modelos[nome] = {
                'accuracy_mean': np.mean(scores_accuracy),
                'accuracy_std': np.std(scores_accuracy),
                'precision_mean': np.mean(scores_precision),
                'precision_std': np.std(scores_precision),
                'recall_mean': np.mean(scores_recall),
                'recall_std': np.std(scores_recall),
                'f1_mean': np.mean(scores_f1),
                'f1_std': np.std(scores_f1)
            }
            
            print(f"  Média: Acc={np.mean(scores_accuracy):.4f}±{np.std(scores_accuracy):.4f}")
        
        self.resultados['melhorias_implementadas'].append("Validação cruzada temporal")
        self.resultados['modelos_otimizados'] = resultados_modelos
        
        return resultados_modelos
    
    def treinar_modelo_final_otimizado(self, X, y, melhor_arquitetura):
        """
        Treina modelo final otimizado
        """
        print("\n=== TREINANDO MODELO FINAL OTIMIZADO ===")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Criar e treinar modelo final
        if melhor_arquitetura == 'neural_network':
            modelo_final = tf.keras.Sequential([
                layers.Input(shape=(X.shape[1],)),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(25, activation='sigmoid')
            ])
            
            modelo_final.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.3, patience=7, min_lr=1e-7)
            ]
            
            history = modelo_final.fit(
                X_train, y_train,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
        # Avaliar modelo final
        y_pred = modelo_final.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Métricas finais
        accuracy = accuracy_score(y_test.flatten(), y_pred_binary.flatten())
        precision = precision_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
        recall = recall_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
        f1 = f1_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted', zero_division=0)
        
        # Taxa de acerto por jogo
        acertos_por_jogo = []
        for i in range(len(y_test)):
            numeros_reais = np.where(y_test[i] == 1)[0] + 1
            numeros_preditos = np.where(y_pred_binary[i] == 1)[0] + 1
            acertos = len(set(numeros_reais) & set(numeros_preditos))
            acertos_por_jogo.append(acertos)
        
        taxa_acerto_media = np.mean(acertos_por_jogo)
        taxa_acerto_percentual = (taxa_acerto_media / 15) * 100
        
        metricas_finais = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'taxa_acerto_media': taxa_acerto_media,
            'taxa_acerto_percentual': taxa_acerto_percentual,
            'distribuicao_acertos': np.bincount(acertos_por_jogo, minlength=16).tolist()
        }
        
        self.resultados['performance_depois'] = metricas_finais
        
        print(f"\n=== RESULTADOS FINAIS ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Taxa de acerto média: {taxa_acerto_media:.2f} números por jogo")
        print(f"Taxa de acerto percentual: {taxa_acerto_percentual:.2f}%")
        
        # Salvar modelo otimizado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modelo_otimizado_path = f"./modelo/saved_models/modelo_otimizado_{timestamp}"
        os.makedirs(modelo_otimizado_path, exist_ok=True)
        
        modelo_final.save(os.path.join(modelo_otimizado_path, "modelo.h5"))
        
        # Salvar métricas
        with open(os.path.join(modelo_otimizado_path, "metricas.json"), 'w') as f:
            json.dump(metricas_finais, f, indent=2)
        
        print(f"Modelo otimizado salvo em: {modelo_otimizado_path}")
        
        return modelo_final, metricas_finais
    
    def gerar_relatorio_completo(self):
        """
        Gera relatório completo dos testes e otimizações
        """
        print("\n=== GERANDO RELATÓRIO COMPLETO ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        relatorio_path = f"./experimentos/resultados/relatorio_otimizacao_{timestamp}.json"
        
        # Calcular melhoria
        if self.resultados['performance_antes'] and self.resultados['performance_depois']:
            melhoria_percentual = (
                self.resultados['performance_depois']['taxa_acerto_percentual'] - 
                self.resultados['performance_antes']['taxa_acerto_percentual']
            )
            self.resultados['melhoria_percentual'] = melhoria_percentual
        
        # Salvar relatório
        os.makedirs(os.path.dirname(relatorio_path), exist_ok=True)
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            json.dump(self.resultados, f, indent=2, ensure_ascii=False)
        
        print(f"Relatório salvo em: {relatorio_path}")
        
        # Resumo no console
        print("\n" + "="*60)
        print("RESUMO DA OTIMIZAÇÃO")
        print("="*60)
        
        if self.resultados['performance_antes']:
            print(f"Performance ANTES: {self.resultados['performance_antes']['taxa_acerto_percentual']:.2f}%")
        
        if self.resultados['performance_depois']:
            print(f"Performance DEPOIS: {self.resultados['performance_depois']['taxa_acerto_percentual']:.2f}%")
        
        if 'melhoria_percentual' in self.resultados:
            print(f"Melhoria: +{self.resultados['melhoria_percentual']:.2f} pontos percentuais")
        
        print(f"\nProblemas identificados: {len(self.resultados['problemas_identificados'])}")
        print(f"Melhorias implementadas: {len(self.resultados['melhorias_implementadas'])}")
        
        return relatorio_path
    
    def executar_teste_completo(self):
        """
        Executa teste completo de performance e otimização
        """
        print("INICIANDO TESTE DE PERFORMANCE REAL DA IA LOTOFÁCIL")
        print("="*60)
        
        try:
            # 1. Carregar dados reais
            dados = self.carregar_dados_reais()
            if dados is None:
                print("Erro: Não foi possível carregar dados")
                return
            
            # 2. Preparar dados
            X, y = self.preparar_dados_para_teste(dados)
            if X is None:
                print("Erro: Não foi possível preparar dados")
                return
            
            # 3. Testar modelo atual
            metricas_atuais, X_train, X_test, y_train, y_test = self.testar_modelo_atual(X, y)
            
            # 4. Identificar problemas
            problemas = self.identificar_problemas_arquitetura(metricas_atuais)
            
            # 5. Implementar melhorias
            X_melhorado = self.implementar_feature_engineering_avancada(X, y)
            
            # 6. Criar modelos otimizados
            modelos_otimizados = self.criar_modelo_ensemble_otimizado(X_melhorado.shape[1])
            
            # 7. Treinar com validação temporal
            resultados_validacao = self.treinar_com_validacao_temporal(X_melhorado, y, modelos_otimizados)
            
            # 8. Selecionar melhor modelo
            melhor_modelo = max(resultados_validacao.keys(), 
                              key=lambda k: resultados_validacao[k]['accuracy_mean'])
            
            print(f"\nMelhor modelo: {melhor_modelo}")
            print(f"Accuracy média: {resultados_validacao[melhor_modelo]['accuracy_mean']:.4f}")
            
            # 9. Treinar modelo final
            modelo_final, metricas_finais = self.treinar_modelo_final_otimizado(
                X_melhorado, y, melhor_modelo
            )
            
            # 10. Gerar relatório
            relatorio_path = self.gerar_relatorio_completo()
            
            print("\nTESTE DE PERFORMANCE CONCLUÍDO COM SUCESSO!")
            return relatorio_path
            
        except Exception as e:
            print(f"Erro durante execução: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    teste = TestePerformanceReal()
    resultado = teste.executar_teste_completo()
    
    if resultado:
        print(f"\nRelatório disponível em: {resultado}")
    else:
        print("\nTeste falhou. Verifique os logs acima.")