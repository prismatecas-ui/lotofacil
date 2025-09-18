import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import os
from dados.dados import dividir_dados


class AlgoritmosAvancados:
    """
    Classe com algoritmos avançados para predição da Lotofácil
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0.0
    
    def create_lstm_model(self, sequence_length=10, input_dim=15):
        """
        Cria modelo LSTM para análise de sequências temporais
        """
        
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, input_dim)),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_cnn_model(self, input_shape=(15, 1)):
        """
        Cria modelo CNN para análise de padrões nos números
        """
        
        model = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_attention_model(self, input_dim=15):
        """
        Cria modelo com mecanismo de atenção
        """
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,))
        
        # Reshape para usar atenção
        reshaped = layers.Reshape((input_dim, 1))(inputs)
        
        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=8
        )(reshaped, reshaped)
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(pooled)
        dropout1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(32, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.2)(dense2)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(dropout2)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_random_forest(self, x_train, y_train, x_test, y_test):
        """
        Treina modelo Random Forest
        """
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(x_train, y_train)
        
        # Predições
        y_pred = rf_model.predict(x_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': rf_model.feature_importances_
        }
        
        return rf_model, accuracy
    
    def train_gradient_boosting(self, x_train, y_train, x_test, y_test):
        """
        Treina modelo Gradient Boosting
        """
        
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        gb_model.fit(x_train, y_train)
        
        # Predições
        y_pred = gb_model.predict(x_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.models['gradient_boosting'] = gb_model
        self.results['gradient_boosting'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': gb_model.feature_importances_
        }
        
        return gb_model, accuracy
    
    def train_svm(self, x_train, y_train, x_test, y_test):
        """
        Treina modelo SVM
        """
        
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        svm_model.fit(x_train, y_train)
        
        # Predições
        y_pred = svm_model.predict(x_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.models['svm'] = svm_model
        self.results['svm'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return svm_model, accuracy
    
    def train_ensemble_model(self, base_dados):
        """
        Treina um ensemble de modelos
        """
        
        x_train, x_test, y_train, y_test, _ = dividir_dados(base_dados)
        
        # Treinar modelos individuais
        print("Treinando Random Forest...")
        self.train_random_forest(x_train, y_train, x_test, y_test)
        
        print("Treinando Gradient Boosting...")
        self.train_gradient_boosting(x_train, y_train, x_test, y_test)
        
        print("Treinando SVM...")
        self.train_svm(x_train, y_train, x_test, y_test)
        
        # Treinar modelos de deep learning
        print("Treinando CNN...")
        cnn_model = self.create_cnn_model()
        x_train_cnn = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test_cnn = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
        cnn_history = cnn_model.fit(
            x_train_cnn, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        cnn_pred = cnn_model.predict(x_test_cnn)
        cnn_pred_binary = (cnn_pred > 0.5).astype(int)
        cnn_accuracy = accuracy_score(y_test, cnn_pred_binary)
        
        self.models['cnn'] = cnn_model
        self.results['cnn'] = {
            'accuracy': cnn_accuracy,
            'precision': precision_score(y_test, cnn_pred_binary, zero_division=0),
            'recall': recall_score(y_test, cnn_pred_binary, zero_division=0),
            'f1_score': f1_score(y_test, cnn_pred_binary, zero_division=0)
        }
        
        print("Treinando modelo com Atenção...")
        attention_model = self.create_attention_model()
        
        attention_history = attention_model.fit(
            x_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        attention_pred = attention_model.predict(x_test)
        attention_pred_binary = (attention_pred > 0.5).astype(int)
        attention_accuracy = accuracy_score(y_test, attention_pred_binary)
        
        self.models['attention'] = attention_model
        self.results['attention'] = {
            'accuracy': attention_accuracy,
            'precision': precision_score(y_test, attention_pred_binary, zero_division=0),
            'recall': recall_score(y_test, attention_pred_binary, zero_division=0),
            'f1_score': f1_score(y_test, attention_pred_binary, zero_division=0)
        }
        
        # Encontrar o melhor modelo
        self.find_best_model()
        
        return self.models, self.results
    
    def find_best_model(self):
        """
        Encontra o modelo com melhor performance
        """
        
        best_score = 0
        best_model_name = None
        
        for model_name, results in self.results.items():
            # Usar F1-score como métrica principal
            score = results['f1_score']
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model = best_model_name
        self.best_score = best_score
        
        print(f"\n=== Melhor Modelo: {best_model_name} ===")
        print(f"F1-Score: {best_score:.4f}")
        
        return best_model_name, best_score
    
    def predict_with_ensemble(self, input_numbers):
        """
        Faz predição usando ensemble de modelos
        """
        
        if len(input_numbers) != 15:
            raise ValueError("É necessário fornecer exatamente 15 números")
        
        input_array = np.array(input_numbers).reshape(1, -1)
        predictions = {}
        
        # Predições dos modelos tradicionais
        for model_name in ['random_forest', 'gradient_boosting', 'svm']:
            if model_name in self.models:
                pred_prob = self.models[model_name].predict_proba(input_array)[0][1]
                predictions[model_name] = pred_prob
        
        # Predições dos modelos de deep learning
        if 'cnn' in self.models:
            input_cnn = input_array.reshape(1, 15, 1)
            pred_cnn = self.models['cnn'].predict(input_cnn)[0][0]
            predictions['cnn'] = pred_cnn
        
        if 'attention' in self.models:
            pred_attention = self.models['attention'].predict(input_array)[0][0]
            predictions['attention'] = pred_attention
        
        # Ensemble por média ponderada
        weights = {
            'random_forest': 0.2,
            'gradient_boosting': 0.2,
            'svm': 0.15,
            'cnn': 0.225,
            'attention': 0.225
        }
        
        ensemble_prob = sum(
            predictions.get(model, 0) * weights.get(model, 0)
            for model in weights.keys()
        )
        
        return {
            'numbers': input_numbers,
            'individual_predictions': predictions,
            'ensemble_probability': float(ensemble_prob),
            'prediction': 'Favorável' if ensemble_prob > 0.5 else 'Desfavorável',
            'confidence': float(abs(ensemble_prob - 0.5) * 2),
            'best_model': self.best_model,
            'best_model_score': self.best_score
        }
    
    def save_models(self, directory="./modelo/algoritmos_avancados"):
        """
        Salva todos os modelos treinados
        """
        
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salvar modelos tradicionais
        for model_name in ['random_forest', 'gradient_boosting', 'svm']:
            if model_name in self.models:
                filename = f"{directory}/{model_name}_{timestamp}.joblib"
                joblib.dump(self.models[model_name], filename)
        
        # Salvar modelos de deep learning
        for model_name in ['cnn', 'attention']:
            if model_name in self.models:
                filename = f"{directory}/{model_name}_{timestamp}.h5"
                self.models[model_name].save(filename)
        
        # Salvar resultados
        results_file = f"{directory}/results_{timestamp}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Resultados dos Algoritmos Avançados\n")
            f.write(f"Data: {datetime.now()}\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"=== {model_name.upper()} ===\n")
                for metric, value in results.items():
                    if metric != 'feature_importance':
                        f.write(f"{metric}: {value}\n")
                f.write("\n")
            
            f.write(f"Melhor Modelo: {self.best_model}\n")
            f.write(f"Melhor Score: {self.best_score}\n")
        
        return directory
    
    def get_model_comparison(self):
        """
        Retorna comparação detalhada dos modelos
        """
        
        comparison = pd.DataFrame(self.results).T
        comparison = comparison.round(4)
        
        # Adicionar ranking
        comparison['rank'] = comparison['f1_score'].rank(ascending=False)
        
        return comparison.sort_values('f1_score', ascending=False)


def treinar_algoritmos_avancados(base_dados):
    """
    Função principal para treinar todos os algoritmos avançados
    """
    
    print("Iniciando treinamento de algoritmos avançados...")
    
    algoritmos = AlgoritmosAvancados()
    models, results = algoritmos.train_ensemble_model(base_dados)
    
    # Salvar modelos
    save_path = algoritmos.save_models()
    print(f"Modelos salvos em: {save_path}")
    
    # Mostrar comparação
    comparison = algoritmos.get_model_comparison()
    print("\n=== Comparação dos Modelos ===")
    print(comparison)
    
    return algoritmos, models, results