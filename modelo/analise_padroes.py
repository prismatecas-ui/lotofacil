import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, Model
from datetime import datetime
import json
import os


class AnalisePadroesLotofacil:
    """
    Classe para análise avançada de padrões na Lotofácil
    """
    
    def __init__(self):
        self.dados = None
        self.padroes = {}
        self.frequencias = {}
        self.clusters = None
        self.pca_model = None
        
    def carregar_dados(self, base_dados):
        """
        Carrega e prepara os dados para análise
        """
        
        self.dados = base_dados.copy()
        
        # Extrair apenas as colunas dos números sorteados
        colunas_numeros = [f'Bola{i:02d}' for i in range(1, 16)]
        
        # Se as colunas não existirem, tentar outras nomenclaturas
        if not all(col in self.dados.columns for col in colunas_numeros):
            # Tentar nomenclatura alternativa
            colunas_numeros = [col for col in self.dados.columns if 'bola' in col.lower() or col.isdigit()]
            if len(colunas_numeros) < 15:
                # Usar as primeiras 15 colunas numéricas
                colunas_numeros = self.dados.select_dtypes(include=[np.number]).columns[:15].tolist()
        
        self.numeros_sorteados = self.dados[colunas_numeros]
        
        return self.numeros_sorteados
    
    def analisar_frequencia_numeros(self):
        """
        Analisa a frequência de cada número
        """
        
        frequencias = Counter()
        
        for _, sorteio in self.numeros_sorteados.iterrows():
            for numero in sorteio:
                if pd.notna(numero):
                    frequencias[int(numero)] += 1
        
        self.frequencias['numeros'] = dict(frequencias)
        
        # Calcular estatísticas
        valores_freq = list(frequencias.values())
        self.frequencias['estatisticas'] = {
            'media': np.mean(valores_freq),
            'mediana': np.median(valores_freq),
            'desvio_padrao': np.std(valores_freq),
            'min': min(valores_freq),
            'max': max(valores_freq)
        }
        
        return self.frequencias['numeros']
    
    def analisar_padroes_paridade(self):
        """
        Analisa padrões de números pares e ímpares
        """
        
        padroes_paridade = []
        
        for _, sorteio in self.numeros_sorteados.iterrows():
            pares = sum(1 for num in sorteio if pd.notna(num) and int(num) % 2 == 0)
            impares = 15 - pares
            padroes_paridade.append((pares, impares))
        
        contador_padroes = Counter(padroes_paridade)
        
        self.padroes['paridade'] = {
            'distribuicao': dict(contador_padroes),
            'mais_comum': contador_padroes.most_common(1)[0],
            'estatisticas': {
                'media_pares': np.mean([p[0] for p in padroes_paridade]),
                'media_impares': np.mean([p[1] for p in padroes_paridade])
            }
        }
        
        return self.padroes['paridade']
    
    def analisar_sequencias_consecutivas(self):
        """
        Analisa sequências de números consecutivos
        """
        
        sequencias = []
        
        for _, sorteio in self.numeros_sorteados.iterrows():
            numeros = sorted([int(num) for num in sorteio if pd.notna(num)])
            
            seq_atual = 1
            max_seq = 1
            total_seq = 0
            
            for i in range(1, len(numeros)):
                if numeros[i] == numeros[i-1] + 1:
                    seq_atual += 1
                    max_seq = max(max_seq, seq_atual)
                else:
                    if seq_atual > 1:
                        total_seq += 1
                    seq_atual = 1
            
            if seq_atual > 1:
                total_seq += 1
            
            sequencias.append({
                'max_consecutivos': max_seq,
                'total_sequencias': total_seq
            })
        
        self.padroes['sequencias'] = {
            'media_max_consecutivos': np.mean([s['max_consecutivos'] for s in sequencias]),
            'media_total_sequencias': np.mean([s['total_sequencias'] for s in sequencias]),
            'distribuicao_max': Counter([s['max_consecutivos'] for s in sequencias])
        }
        
        return self.padroes['sequencias']
    
    def analisar_soma_numeros(self):
        """
        Analisa a soma dos números sorteados
        """
        
        somas = []
        
        for _, sorteio in self.numeros_sorteados.iterrows():
            soma = sum(int(num) for num in sorteio if pd.notna(num))
            somas.append(soma)
        
        self.padroes['somas'] = {
            'media': np.mean(somas),
            'mediana': np.median(somas),
            'desvio_padrao': np.std(somas),
            'min': min(somas),
            'max': max(somas),
            'quartis': np.percentile(somas, [25, 50, 75]).tolist()
        }
        
        return self.padroes['somas']
    
    def analisar_distribuicao_dezenas(self):
        """
        Analisa distribuição por dezenas (1-5, 6-10, 11-15, 16-20, 21-25)
        """
        
        distribuicoes = []
        
        for _, sorteio in self.numeros_sorteados.iterrows():
            dezenas = [0, 0, 0, 0, 0]  # 1-5, 6-10, 11-15, 16-20, 21-25
            
            for num in sorteio:
                if pd.notna(num):
                    numero = int(num)
                    if 1 <= numero <= 5:
                        dezenas[0] += 1
                    elif 6 <= numero <= 10:
                        dezenas[1] += 1
                    elif 11 <= numero <= 15:
                        dezenas[2] += 1
                    elif 16 <= numero <= 20:
                        dezenas[3] += 1
                    elif 21 <= numero <= 25:
                        dezenas[4] += 1
            
            distribuicoes.append(dezenas)
        
        # Calcular médias por dezena
        medias_dezenas = np.mean(distribuicoes, axis=0)
        
        self.padroes['dezenas'] = {
            'media_por_dezena': medias_dezenas.tolist(),
            'distribuicoes': distribuicoes,
            'padroes_comuns': Counter([tuple(d) for d in distribuicoes]).most_common(10)
        }
        
        return self.padroes['dezenas']
    
    def criar_clusters_sorteios(self, n_clusters=8):
        """
        Cria clusters de sorteios similares
        """
        
        # Preparar dados para clustering
        dados_clustering = []
        
        for _, sorteio in self.numeros_sorteados.iterrows():
            # Criar vetor de características
            numeros = [int(num) for num in sorteio if pd.notna(num)]
            
            # Características: frequência de cada número (1-25)
            vetor = [0] * 25
            for num in numeros:
                if 1 <= num <= 25:
                    vetor[num-1] = 1
            
            # Adicionar características extras
            pares = sum(1 for num in numeros if num % 2 == 0)
            soma = sum(numeros)
            
            vetor.extend([pares, soma])
            dados_clustering.append(vetor)
        
        # Normalizar dados
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(dados_clustering)
        
        # Aplicar PCA para redução de dimensionalidade
        self.pca_model = PCA(n_components=10)
        dados_pca = self.pca_model.fit_transform(dados_normalizados)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(dados_pca)
        
        self.clusters = {
            'labels': clusters,
            'centers': kmeans.cluster_centers_,
            'model': kmeans,
            'scaler': scaler,
            'dados_pca': dados_pca
        }
        
        # Analisar características de cada cluster
        self.analisar_clusters()
        
        return clusters
    
    def analisar_clusters(self):
        """
        Analisa as características de cada cluster
        """
        
        if self.clusters is None:
            return None
        
        clusters_info = {}
        
        for cluster_id in range(len(self.clusters['centers'])):
            # Sorteios deste cluster
            indices_cluster = np.where(self.clusters['labels'] == cluster_id)[0]
            sorteios_cluster = self.numeros_sorteados.iloc[indices_cluster]
            
            # Calcular características médias
            numeros_freq = Counter()
            somas = []
            pares_count = []
            
            for _, sorteio in sorteios_cluster.iterrows():
                numeros = [int(num) for num in sorteio if pd.notna(num)]
                
                for num in numeros:
                    numeros_freq[num] += 1
                
                somas.append(sum(numeros))
                pares_count.append(sum(1 for num in numeros if num % 2 == 0))
            
            clusters_info[cluster_id] = {
                'tamanho': len(indices_cluster),
                'numeros_mais_frequentes': numeros_freq.most_common(10),
                'soma_media': np.mean(somas),
                'pares_medio': np.mean(pares_count),
                'indices_sorteios': indices_cluster.tolist()
            }
        
        self.padroes['clusters'] = clusters_info
        
        return clusters_info
    
    def criar_modelo_predicao_padroes(self):
        """
        Cria modelo neural para predição baseada em padrões
        """
        
        # Preparar dados de entrada
        X = []
        y = []
        
        for i in range(len(self.numeros_sorteados) - 1):
            # Usar sorteio atual para predizer próximo
            sorteio_atual = self.numeros_sorteados.iloc[i]
            proximo_sorteio = self.numeros_sorteados.iloc[i + 1]
            
            # Criar vetor de características do sorteio atual
            numeros_atual = [int(num) for num in sorteio_atual if pd.notna(num)]
            
            # Vetor binário para números 1-25
            vetor_atual = [0] * 25
            for num in numeros_atual:
                if 1 <= num <= 25:
                    vetor_atual[num-1] = 1
            
            # Adicionar características extras
            pares = sum(1 for num in numeros_atual if num % 2 == 0)
            soma = sum(numeros_atual)
            
            vetor_atual.extend([pares/15, soma/375])  # Normalizar
            
            X.append(vetor_atual)
            
            # Target: vetor binário do próximo sorteio
            numeros_proximo = [int(num) for num in proximo_sorteio if pd.notna(num)]
            vetor_proximo = [0] * 25
            for num in numeros_proximo:
                if 1 <= num <= 25:
                    vetor_proximo[num-1] = 1
            
            y.append(vetor_proximo)
        
        X = np.array(X)
        y = np.array(y)
        
        # Criar modelo
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(27,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(25, activation='sigmoid')  # Probabilidade para cada número
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Treinar modelo
        history = model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.modelo_padroes = model
        
        return model, history
    
    def predizer_proximos_numeros(self, ultimo_sorteio, top_n=15):
        """
        Prediz os próximos números baseado em padrões
        """
        
        if not hasattr(self, 'modelo_padroes'):
            raise ValueError("Modelo de padrões não foi treinado")
        
        # Preparar entrada
        numeros_atual = [int(num) for num in ultimo_sorteio if pd.notna(num)]
        
        vetor_atual = [0] * 25
        for num in numeros_atual:
            if 1 <= num <= 25:
                vetor_atual[num-1] = 1
        
        pares = sum(1 for num in numeros_atual if num % 2 == 0)
        soma = sum(numeros_atual)
        
        vetor_atual.extend([pares/15, soma/375])
        
        # Fazer predição
        entrada = np.array([vetor_atual])
        probabilidades = self.modelo_padroes.predict(entrada)[0]
        
        # Selecionar top números
        indices_ordenados = np.argsort(probabilidades)[::-1]
        numeros_sugeridos = [(i+1, probabilidades[i]) for i in indices_ordenados[:top_n]]
        
        return numeros_sugeridos
    
    def gerar_relatorio_completo(self):
        """
        Gera relatório completo da análise de padrões
        """
        
        # Executar todas as análises
        self.analisar_frequencia_numeros()
        self.analisar_padroes_paridade()
        self.analisar_sequencias_consecutivas()
        self.analisar_soma_numeros()
        self.analisar_distribuicao_dezenas()
        self.criar_clusters_sorteios()
        
        # Criar relatório
        relatorio = {
            'data_analise': datetime.now().isoformat(),
            'total_sorteios': len(self.numeros_sorteados),
            'frequencias': self.frequencias,
            'padroes': self.padroes,
            'insights': self.gerar_insights()
        }
        
        return relatorio
    
    def gerar_insights(self):
        """
        Gera insights baseados na análise
        """
        
        insights = []
        
        # Insights de frequência
        if 'numeros' in self.frequencias:
            freq_nums = self.frequencias['numeros']
            mais_frequente = max(freq_nums, key=freq_nums.get)
            menos_frequente = min(freq_nums, key=freq_nums.get)
            
            insights.append(f"Número mais sorteado: {mais_frequente} ({freq_nums[mais_frequente]} vezes)")
            insights.append(f"Número menos sorteado: {menos_frequente} ({freq_nums[menos_frequente]} vezes)")
        
        # Insights de paridade
        if 'paridade' in self.padroes:
            padrao_comum = self.padroes['paridade']['mais_comum']
            insights.append(f"Padrão de paridade mais comum: {padrao_comum[0][0]} pares e {padrao_comum[0][1]} ímpares")
        
        # Insights de soma
        if 'somas' in self.padroes:
            soma_media = self.padroes['somas']['media']
            insights.append(f"Soma média dos sorteios: {soma_media:.1f}")
        
        # Insights de clusters
        if 'clusters' in self.padroes:
            maior_cluster = max(self.padroes['clusters'].values(), key=lambda x: x['tamanho'])
            insights.append(f"Maior cluster tem {maior_cluster['tamanho']} sorteios")
        
        return insights
    
    def salvar_relatorio(self, filepath=None):
        """
        Salva o relatório em arquivo JSON
        """
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"./modelo/analise_padroes_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        relatorio = self.gerar_relatorio_completo()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath


def analisar_padroes_lotofacil(base_dados):
    """
    Função principal para análise completa de padrões
    """
    
    print("Iniciando análise de padrões da Lotofácil...")
    
    analise = AnalisePadroesLotofacil()
    analise.carregar_dados(base_dados)
    
    # Gerar relatório completo
    relatorio = analise.gerar_relatorio_completo()
    
    # Salvar relatório
    filepath = analise.salvar_relatorio()
    print(f"Relatório salvo em: {filepath}")
    
    # Treinar modelo de predição
    try:
        modelo, history = analise.criar_modelo_predicao_padroes()
        print("Modelo de predição de padrões treinado com sucesso")
    except Exception as e:
        print(f"Erro ao treinar modelo de padrões: {e}")
    
    return analise, relatorio