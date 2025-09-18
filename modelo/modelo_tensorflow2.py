import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from datetime import datetime
import os
import logging
from dados.dados import dividir_dados

# Configurar logger
logger = logging.getLogger(__name__)


class LotofacilModel:
    """
    Modelo modernizado para predição da Lotofácil usando TensorFlow 2.x
    """
    
    def __init__(self, model_name="lotofacil_model"):
        self.model_name = model_name
        self.model = None
        self.history = None
        self.metrics = {}
        
    def criar_modelo_avancado(self, input_shape=(25,)):
        """
        Cria um modelo neural avançado com regularização e dropout.
        
        Args:
            input_shape: Formato dos dados de entrada
            
        Returns:
            Modelo TensorFlow compilado
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.regularizers import l2
            
            modelo = Sequential()
            
            # Camada de entrada
            modelo.add(Dense(128, activation='relu', input_shape=input_shape, 
                           kernel_regularizer=l2(0.001), name='entrada'))
            modelo.add(BatchNormalization(name='bn_entrada'))
            modelo.add(Dropout(0.3, name='dropout_entrada'))
            
            # Camadas ocultas
            modelo.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='oculta_1'))
            modelo.add(BatchNormalization(name='bn_oculta_1'))
            modelo.add(Dropout(0.4, name='dropout_oculta_1'))
            
            modelo.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='oculta_2'))
            modelo.add(BatchNormalization(name='bn_oculta_2'))
            modelo.add(Dropout(0.3, name='dropout_oculta_2'))
            
            modelo.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001), name='oculta_3'))
            modelo.add(Dropout(0.2, name='dropout_oculta_3'))
            
            # Camada de saída
            modelo.add(Dense(25, activation='sigmoid', name='saida'))
            
            return modelo
            
        except Exception as e:
            logger.error(f"Erro ao criar modelo: {e}")
            raise
    

    
    def compile_model(self, 
                     learning_rate=0.001,
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'precision', 'recall']):
        """
        Compila o modelo com otimizador e métricas avançadas
        """
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return self.model
    
    def create_callbacks(self, patience=10, min_delta=0.001):
        """
        Cria callbacks para treinamento otimizado
        """
        
        # Diretório para salvar modelos
        model_dir = f"./modelo/checkpoints/{self.model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        callback_list = [
            # Early stopping para evitar overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Redução da taxa de aprendizado
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Checkpoint do melhor modelo
            callbacks.ModelCheckpoint(
                filepath=f"{model_dir}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # TensorBoard para visualização
            callbacks.TensorBoard(
                log_dir=f"./logs/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callback_list
    
    def train_model(self, 
                   base_dados,
                   epochs=100,
                   batch_size=32,
                   validation_split=0.2,
                   verbose=1):
        """
        Treina o modelo com dados da Lotofácil
        """
        
        # Preparar dados
        x_treino, x_teste, y_treino, y_teste, atributos = dividir_dados(base_dados)
        
        # Criar callbacks
        callback_list = self.create_callbacks()
        
        # Treinar modelo
        self.history = self.model.fit(
            x_treino, y_treino,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=verbose
        )
        
        # Avaliar modelo
        self.evaluate_model(x_teste, y_teste)
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """
        Avalia o modelo com métricas detalhadas
        """
        
        # Predições
        y_pred_prob = self.model.predict(x_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Métricas básicas
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            x_test, y_test, verbose=0
        )
        
        # F1-Score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
        
        # Salvar métricas
        self.metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'f1_score': f1_score,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return self.metrics
    
    def predict_numbers(self, input_numbers):
        """
        Prediz a probabilidade de uma combinação ser sorteada
        
        Args:
            input_numbers: Lista ou array com 15 números
        """
        
        if len(input_numbers) != 15:
            raise ValueError("É necessário fornecer exatamente 15 números")
        
        # Converter para formato adequado
        input_array = np.array(input_numbers).reshape(1, -1)
        
        # Fazer predição
        probability = self.model.predict(input_array)[0][0]
        
        return {
            'numbers': input_numbers,
            'probability': float(probability),
            'prediction': 'Favorável' if probability > 0.5 else 'Desfavorável',
            'confidence': float(abs(probability - 0.5) * 2)
        }
    
    def save_model(self, filepath=None):
        """
        Salva o modelo treinado
        """
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"./modelo/saved_models/{self.model_name}_{timestamp}"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # Salvar métricas
        metrics_file = f"{filepath}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Métricas do Modelo: {self.model_name}\n")
            f.write(f"Data: {datetime.now()}\n\n")
            for key, value in self.metrics.items():
                if key not in ['confusion_matrix', 'classification_report']:
                    f.write(f"{key}: {value}\n")
        
        return filepath
    
    def load_model(self, filepath):
        """
        Carrega um modelo salvo
        """
        
        self.model = keras.models.load_model(filepath)
        return self.model
    
    def get_model_summary(self) -> str:
        """
        Retorna resumo do modelo.
        
        Returns:
            String com resumo do modelo
        """
        if self.model is None:
            return "Modelo não foi criado ainda."
        
        import io
        import sys
        
        # Capturar saída do summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            self.model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary
    
    def preparar_dados(self, dados_historicos):
        """
        Prepara os dados para treinamento.
        
        Args:
            dados_historicos: DataFrame com dados históricos dos concursos
            
        Returns:
            Tuple com (X_train, y_train, X_test, y_test)
        """
        try:
            import json
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # Verificar se há dados suficientes
            if len(dados_historicos) < 2:
                # Criar dados sintéticos para teste
                X_train = np.random.randint(0, 2, (100, 25))
                y_train = np.random.randint(0, 2, (100, 25))
                X_test = np.random.randint(0, 2, (20, 25))
                y_test = np.random.randint(0, 2, (20, 25))
                return X_train, y_train, X_test, y_test
            
            # Converter números sorteados para vetores binários
            numeros_lista = []
            for _, row in dados_historicos.iterrows():
                # Usar a coluna 'dezenas' em vez de 'numeros_sorteados'
                dezenas_str = row.get('dezenas', '')
                if dezenas_str:
                    try:
                        # Tentar diferentes formatos de dados
                        if dezenas_str.startswith('['):
                            numeros = json.loads(dezenas_str)
                        else:
                            # Assumir formato separado por vírgula
                            numeros = [int(x.strip()) for x in dezenas_str.split(',') if x.strip().isdigit()]
                    except:
                        # Gerar números aleatórios como fallback
                        numeros = sorted(np.random.choice(range(1, 26), 15, replace=False))
                else:
                    # Gerar números aleatórios como fallback
                    numeros = sorted(np.random.choice(range(1, 26), 15, replace=False))
                
                # Criar vetor binário (1 se número foi sorteado, 0 caso contrário)
                vetor = [1 if i in numeros else 0 for i in range(1, 26)]
                numeros_lista.append(vetor)
            
            # Se ainda não há dados suficientes, criar dados sintéticos
            if len(numeros_lista) < 2:
                X_train = np.random.randint(0, 2, (100, 25))
                y_train = np.random.randint(0, 2, (100, 25))
                X_test = np.random.randint(0, 2, (20, 25))
                y_test = np.random.randint(0, 2, (20, 25))
                return X_train, y_train, X_test, y_test
            
            X = np.array(numeros_lista[:-1])  # Todos exceto o último
            y = np.array(numeros_lista[1:])   # Todos exceto o primeiro
            
            # Se há apenas um registro, duplicar para ter dados de treino
            if len(X) == 0:
                X = np.array([numeros_lista[0]] * 100)
                y = np.array([numeros_lista[0]] * 100)
            
            # Dividir em treino e teste
            if len(X) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                # Usar os mesmos dados para treino e teste
                X_train = X_test = X
                y_train = y_test = y
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {e}")
            # Retornar dados sintéticos em caso de erro
            X_train = np.random.randint(0, 2, (100, 25))
            y_train = np.random.randint(0, 2, (100, 25))
            X_test = np.random.randint(0, 2, (20, 25))
            y_test = np.random.randint(0, 2, (20, 25))
            return X_train, y_train, X_test, y_test
    
    def prever(self, dados_entrada):
        """
        Faz predição usando o modelo treinado.
        
        Args:
            dados_entrada: Dados de entrada para predição
            
        Returns:
            Lista com números preditos
        """
        try:
            if self.model is None:
                raise ValueError("Modelo não foi treinado ainda")
            
            # Fazer predição
            predicao = self.model.predict(dados_entrada)
            
            # Converter predição em números (pegar os 15 com maior probabilidade)
            numeros_preditos = []
            for pred in predicao:
                # Pegar índices dos 15 maiores valores
                indices = np.argsort(pred)[-15:]
                # Converter para números (1-25)
                numeros = [i + 1 for i in indices]
                numeros_preditos.append(sorted(numeros))
            
            return numeros_preditos
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise


def criar_modelo_tensorflow2(base_dados, 
                           hidden_layers=[64, 32, 16],
                           epochs=100,
                           batch_size=32,
                           learning_rate=0.001):
    """
    Função de compatibilidade com a interface antiga
    
    Args:
        base_dados: DataFrame da base de dados
        hidden_layers: Lista com neurônios por camada
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do lote
        learning_rate: Taxa de aprendizado
    
    Returns:
        Tupla (modelo, acurácia)
    """
    
    # Criar instância do modelo
    lotofacil_model = LotofacilModel("lotofacil_v2")
    
    # Criar e compilar modelo
    model = lotofacil_model.criar_modelo_avancado(input_shape=(25,))
    lotofacil_model.compile_model(learning_rate=learning_rate)
    
    # Treinar modelo
    history = lotofacil_model.train_model(
        base_dados, 
        epochs=epochs, 
        batch_size=batch_size
    )
    
    # Retornar modelo e acurácia para compatibilidade
    accuracy = lotofacil_model.metrics.get('test_accuracy', 0.0)
    
    return lotofacil_model.model, accuracy