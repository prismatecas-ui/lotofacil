from dados.dados import dividir_dados

# Atualizado para TensorFlow 2.x
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


def criar_modelo(
                    base_dados, 
                    primeira_camada=64,
                    segunda_camada=32,
                    terceira_camada=16,
                    saida=1,
                    periodo=100,
                    lote=32,
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    use_callbacks=True
                ):
    """
    Cria o modelo sequencial com três camadas usando TensorFlow 2.x.

    :param base_dados: DataFrame da base de dados.
    :param primeira_camada: camada de entrada utilizando a função retificadora (relu). Default: 64 neurônios.
    :param segunda_camada: segunda camada utilizando a função retificadora(relu). Default: 32 neurônios.
    :param terceira_camada: terceira camada utilizando a função retificadora (relu). Default: 16 neurônios.
    :param saida: camada de saída utilizando a função de ativação (sigmoid). Default: 1 neurônio.
    :param periodo: quantidade de épocas para ajuste do modelo. Default: 100.
    :param lote: tamanho do lote para treinamento. Default: 32.
    :param dropout_rate: taxa de dropout para regularização. Default: 0.2.
    :param learning_rate: taxa de aprendizado. Default: 0.001.
    :param use_callbacks: usar callbacks para otimização. Default: True.
    :return: tupla (modelo, acurácia, histórico).
    """

    x_treino, x_teste, y_treino, y_teste, atributos = dividir_dados(base_dados)

    # Criando o modelo com TensorFlow 2.x
    modelo = Sequential([
        Dense(primeira_camada, input_dim=atributos, activation='relu', name='input_layer'),
        Dropout(dropout_rate, name='dropout_1'),
        Dense(segunda_camada, activation='relu', name='hidden_layer_1'),
        Dropout(dropout_rate, name='dropout_2'),
        Dense(terceira_camada, activation='relu', name='hidden_layer_2'),
        Dropout(dropout_rate, name='dropout_3'),
        Dense(saida, activation='sigmoid', name='output_layer')
    ])

    # Compilando o modelo com otimizador Adam personalizado
    optimizer = Adam(learning_rate=learning_rate)
    modelo.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'precision', 'recall']
    )

    # Preparando callbacks se solicitado
    callbacks_list = []
    if use_callbacks:
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

    # Treinando o modelo com validação
    history = modelo.fit(
        x_treino, y_treino,
        epochs=periodo,
        batch_size=lote,
        validation_split=0.2,
        callbacks=callbacks_list,
        verbose=1
    )

    # Avaliação detalhada do modelo
    pontuacao = modelo.evaluate(x_teste, y_teste, verbose=0)
    
    # Predições para métricas adicionais
    y_pred = modelo.predict(x_teste)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculando F1-Score
    from sklearn.metrics import f1_score, classification_report
    f1 = f1_score(y_teste, y_pred_binary)
    
    print(f"\n=== Resultados do Modelo ===")
    print(f"Acurácia: {pontuacao[1]:.4f}")
    print(f"Precisão: {pontuacao[2]:.4f}")
    print(f"Recall: {pontuacao[3]:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Loss: {pontuacao[0]:.4f}")
    
    return modelo, pontuacao[1], history

