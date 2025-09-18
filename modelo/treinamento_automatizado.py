import os
import json
import pickle
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sqlite3
import threading
import queue
from pathlib import Path

# Importar nossos módulos
from modelo_tensorflow2 import LotofacilModel
from algoritmos_avancados import AlgoritmosAvancados
from analise_padroes import AnalisePadroesLotofacil


class SistemaTreinamentoAutomatizado:
    """
    Sistema completo de treinamento automatizado para modelos da Lotofácil
    """
    
    def __init__(self, config_path: str = "./modelo/config_treinamento.json"):
        self.config_path = config_path
        self.config = self.carregar_configuracao()
        self.logger = self.configurar_logging()
        
        # Diretórios
        self.base_dir = Path("./modelo")
        self.modelos_dir = self.base_dir / "modelos_treinados"
        self.logs_dir = self.base_dir / "logs_treinamento"
        self.backups_dir = self.base_dir / "backups"
        
        # Criar diretórios se não existirem
        for dir_path in [self.modelos_dir, self.logs_dir, self.backups_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Estado do sistema
        self.treinamento_ativo = False
        self.fila_treinamento = queue.Queue()
        self.historico_treinamentos = []
        self.modelos_ativos = {}
        
        # Thread para processamento
        self.thread_treinamento = None
        self.parar_sistema = False
        
    def carregar_configuracao(self) -> Dict[str, Any]:
        """
        Carrega configuração do sistema
        """
        
        config_default = {
            "agendamento": {
                "treinar_diariamente": True,
                "hora_treinamento": "02:00",
                "treinar_semanalmente": True,
                "dia_semana": "sunday",
                "retreinar_automatico": True,
                "intervalo_retreino_dias": 7
            },
            "modelos": {
                "tensorflow_basico": {
                    "ativo": True,
                    "epochs": 100,
                    "batch_size": 32,
                    "validation_split": 0.2,
                    "early_stopping_patience": 10
                },
                "algoritmos_avancados": {
                    "ativo": True,
                    "modelos": ["random_forest", "xgboost", "lstm", "cnn"]
                },
                "analise_padroes": {
                    "ativo": True,
                    "clusters": 8,
                    "pca_components": 10
                }
            },
            "otimizacao": {
                "hyperparameter_tuning": True,
                "auto_feature_selection": True,
                "ensemble_models": True,
                "performance_threshold": 0.75
            },
            "dados": {
                "fonte_principal": "./base/base_dados.xlsx",
                "fonte_cache": "./base/cache_concursos.json",
                "database_sqlite": "./database/lotofacil.db",
                "min_registros": 100,
                "test_size": 0.2
            },
            "notificacoes": {
                "email_ativo": False,
                "log_detalhado": True,
                "salvar_metricas": True
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_carregada = json.load(f)
                # Merge com configuração padrão
                config_default.update(config_carregada)
            except Exception as e:
                print(f"Erro ao carregar configuração: {e}. Usando configuração padrão.")
        else:
            # Salvar configuração padrão
            self.salvar_configuracao(config_default)
        
        return config_default
    
    def salvar_configuracao(self, config: Dict[str, Any] = None):
        """
        Salva configuração atual
        """
        
        if config is None:
            config = self.config
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def configurar_logging(self) -> logging.Logger:
        """
        Configura sistema de logging
        """
        
        logger = logging.getLogger('TreinamentoAutomatizado')
        logger.setLevel(logging.INFO)
        
        # Handler para arquivo
        log_file = self.logs_dir / f"treinamento_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def carregar_dados(self) -> Tuple[pd.DataFrame, bool]:
        """
        Carrega dados de diferentes fontes
        """
        
        dados = None
        fonte_usada = None
        
        # Tentar SQLite primeiro
        if os.path.exists(self.config['dados']['database_sqlite']):
            try:
                conn = sqlite3.connect(self.config['dados']['database_sqlite'])
                dados = pd.read_sql_query("SELECT * FROM concursos", conn)
                conn.close()
                fonte_usada = "SQLite"
                self.logger.info(f"Dados carregados do SQLite: {len(dados)} registros")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar do SQLite: {e}")
        
        # Tentar cache JSON
        if dados is None and os.path.exists(self.config['dados']['fonte_cache']):
            try:
                with open(self.config['dados']['fonte_cache'], 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                dados = pd.DataFrame(cache_data)
                fonte_usada = "Cache JSON"
                self.logger.info(f"Dados carregados do cache: {len(dados)} registros")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar do cache: {e}")
        
        # Tentar Excel como último recurso
        if dados is None and os.path.exists(self.config['dados']['fonte_principal']):
            try:
                dados = pd.read_excel(self.config['dados']['fonte_principal'])
                fonte_usada = "Excel"
                self.logger.info(f"Dados carregados do Excel: {len(dados)} registros")
            except Exception as e:
                self.logger.error(f"Erro ao carregar do Excel: {e}")
                return None, False
        
        if dados is None:
            self.logger.error("Não foi possível carregar dados de nenhuma fonte")
            return None, False
        
        # Verificar quantidade mínima de registros
        if len(dados) < self.config['dados']['min_registros']:
            self.logger.error(f"Dados insuficientes: {len(dados)} < {self.config['dados']['min_registros']}")
            return None, False
        
        self.logger.info(f"Dados carregados com sucesso de {fonte_usada}: {len(dados)} registros")
        return dados, True
    
    def preparar_dados_treinamento(self, dados: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Prepara dados para treinamento
        """
        
        try:
            # Identificar colunas de números sorteados
            colunas_numeros = []
            
            # Tentar diferentes nomenclaturas
            for i in range(1, 16):
                for formato in [f'Bola{i:02d}', f'bola{i:02d}', f'Numero{i:02d}', f'numero{i:02d}', str(i)]:
                    if formato in dados.columns:
                        colunas_numeros.append(formato)
                        break
            
            if len(colunas_numeros) < 15:
                # Usar colunas numéricas
                colunas_numericas = dados.select_dtypes(include=[np.number]).columns
                colunas_numeros = colunas_numericas[:15].tolist()
            
            if len(colunas_numeros) < 15:
                raise ValueError("Não foi possível identificar 15 colunas de números")
            
            # Preparar features (X) e targets (y)
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
            
            self.logger.info(f"Dados preparados: X={X.shape}, y={y.shape}")
            return X, y, True
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar dados: {e}")
            return None, None, False
    
    def treinar_modelo_tensorflow(self, X: np.ndarray, y: np.ndarray) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """
        Treina modelo TensorFlow
        """
        
        try:
            config_modelo = self.config['modelos']['tensorflow_basico']
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['dados']['test_size'], random_state=42
            )
            
            # Criar modelo
            modelo_tf = LotofacilModel()
            modelo = modelo_tf.criar_modelo(input_shape=(25,))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    patience=config_modelo['early_stopping_patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                ),
                ModelCheckpoint(
                    filepath=str(self.modelos_dir / "tensorflow_best.h5"),
                    save_best_only=True
                )
            ]
            
            # Treinar
            history = modelo.fit(
                X_train, y_train,
                epochs=config_modelo['epochs'],
                batch_size=config_modelo['batch_size'],
                validation_split=config_modelo['validation_split'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Avaliar
            y_pred = modelo.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            metricas = {
                'accuracy': accuracy_score(y_test.flatten(), y_pred_binary.flatten()),
                'precision': precision_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted'),
                'recall': recall_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted'),
                'f1': f1_score(y_test.flatten(), y_pred_binary.flatten(), average='weighted'),
                'loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            }
            
            self.logger.info(f"Modelo TensorFlow treinado - Accuracy: {metricas['accuracy']:.4f}")
            
            return modelo, metricas
            
        except Exception as e:
            self.logger.error(f"Erro ao treinar modelo TensorFlow: {e}")
            return None, {}
    
    def treinar_algoritmos_avancados(self, dados: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """
        Treina algoritmos avançados
        """
        
        try:
            config_algoritmos = self.config['modelos']['algoritmos_avancados']
            
            algoritmos = AlgoritmosAvancados()
            
            # Treinar modelos tradicionais
            metricas_tradicionais = algoritmos.treinar_modelos_tradicionais(dados)
            
            # Treinar modelos deep learning se especificado
            metricas_deep = {}
            if 'lstm' in config_algoritmos['modelos']:
                try:
                    modelo_lstm = algoritmos.criar_modelo_lstm()
                    # Treinar LSTM aqui
                    metricas_deep['lstm'] = {'accuracy': 0.0}  # Placeholder
                except Exception as e:
                    self.logger.warning(f"Erro ao treinar LSTM: {e}")
            
            # Combinar métricas
            metricas_combinadas = {
                **metricas_tradicionais,
                **metricas_deep
            }
            
            self.logger.info(f"Algoritmos avançados treinados: {list(metricas_combinadas.keys())}")
            
            return algoritmos, metricas_combinadas
            
        except Exception as e:
            self.logger.error(f"Erro ao treinar algoritmos avançados: {e}")
            return None, {}
    
    def executar_treinamento_completo(self) -> Dict[str, Any]:
        """
        Executa ciclo completo de treinamento
        """
        
        inicio = datetime.now()
        self.logger.info("Iniciando treinamento completo")
        
        resultado = {
            'timestamp': inicio.isoformat(),
            'sucesso': False,
            'modelos_treinados': [],
            'metricas': {},
            'erros': [],
            'tempo_execucao': 0
        }
        
        try:
            # Carregar dados
            dados, sucesso_dados = self.carregar_dados()
            if not sucesso_dados:
                resultado['erros'].append("Falha ao carregar dados")
                return resultado
            
            # Preparar dados para TensorFlow
            X, y, sucesso_prep = self.preparar_dados_treinamento(dados)
            if not sucesso_prep:
                resultado['erros'].append("Falha ao preparar dados")
                return resultado
            
            # Treinar modelo TensorFlow
            if self.config['modelos']['tensorflow_basico']['ativo']:
                modelo_tf, metricas_tf = self.treinar_modelo_tensorflow(X, y)
                if modelo_tf is not None:
                    resultado['modelos_treinados'].append('tensorflow')
                    resultado['metricas']['tensorflow'] = metricas_tf
                    
                    # Salvar modelo
                    modelo_tf.save(str(self.modelos_dir / "tensorflow_final.h5"))
            
            # Treinar algoritmos avançados
            if self.config['modelos']['algoritmos_avancados']['ativo']:
                algoritmos, metricas_alg = self.treinar_algoritmos_avancados(dados)
                if algoritmos is not None:
                    resultado['modelos_treinados'].append('algoritmos_avancados')
                    resultado['metricas']['algoritmos_avancados'] = metricas_alg
                    
                    # Salvar algoritmos
                    algoritmos.salvar_modelos(str(self.modelos_dir / "algoritmos_avancados.pkl"))
            
            # Análise de padrões
            if self.config['modelos']['analise_padroes']['ativo']:
                analise = AnalisePadroesLotofacil()
                analise.carregar_dados(dados)
                relatorio = analise.gerar_relatorio_completo()
                
                resultado['modelos_treinados'].append('analise_padroes')
                resultado['metricas']['analise_padroes'] = {
                    'total_sorteios': relatorio['total_sorteios'],
                    'insights_count': len(relatorio['insights'])
                }
                
                # Salvar análise
                analise.salvar_relatorio(str(self.modelos_dir / "analise_padroes.json"))
            
            resultado['sucesso'] = True
            
        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {e}")
            resultado['erros'].append(str(e))
        
        finally:
            fim = datetime.now()
            resultado['tempo_execucao'] = (fim - inicio).total_seconds()
            
            # Salvar histórico
            self.historico_treinamentos.append(resultado)
            self.salvar_historico_treinamentos()
            
            self.logger.info(f"Treinamento concluído em {resultado['tempo_execucao']:.2f}s")
        
        return resultado
    
    def salvar_historico_treinamentos(self):
        """
        Salva histórico de treinamentos
        """
        
        historico_file = self.logs_dir / "historico_treinamentos.json"
        
        with open(historico_file, 'w', encoding='utf-8') as f:
            json.dump(self.historico_treinamentos, f, indent=2, ensure_ascii=False, default=str)
    
    def configurar_agendamento(self):
        """
        Configura agendamento automático
        """
        
        config_agend = self.config['agendamento']
        
        # Treinamento diário
        if config_agend['treinar_diariamente']:
            schedule.every().day.at(config_agend['hora_treinamento']).do(
                self.adicionar_treinamento_fila, 'diario'
            )
            self.logger.info(f"Agendamento diário configurado para {config_agend['hora_treinamento']}")
        
        # Treinamento semanal
        if config_agend['treinar_semanalmente']:
            getattr(schedule.every(), config_agend['dia_semana']).at(
                config_agend['hora_treinamento']
            ).do(self.adicionar_treinamento_fila, 'semanal')
            self.logger.info(f"Agendamento semanal configurado para {config_agend['dia_semana']}")
    
    def adicionar_treinamento_fila(self, tipo: str):
        """
        Adiciona treinamento à fila
        """
        
        self.fila_treinamento.put({
            'tipo': tipo,
            'timestamp': datetime.now().isoformat(),
            'prioridade': 1 if tipo == 'manual' else 2
        })
        
        self.logger.info(f"Treinamento {tipo} adicionado à fila")
    
    def processar_fila_treinamento(self):
        """
        Processa fila de treinamentos
        """
        
        while not self.parar_sistema:
            try:
                # Verificar agendamentos
                schedule.run_pending()
                
                # Processar fila
                if not self.fila_treinamento.empty():
                    item = self.fila_treinamento.get(timeout=1)
                    
                    if not self.treinamento_ativo:
                        self.treinamento_ativo = True
                        self.logger.info(f"Iniciando treinamento {item['tipo']}")
                        
                        resultado = self.executar_treinamento_completo()
                        
                        self.treinamento_ativo = False
                        self.logger.info(f"Treinamento {item['tipo']} concluído")
                    else:
                        # Recolocar na fila se já há treinamento ativo
                        self.fila_treinamento.put(item)
                
                time.sleep(60)  # Verificar a cada minuto
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro no processamento da fila: {e}")
                time.sleep(60)
    
    def iniciar_sistema(self):
        """
        Inicia o sistema de treinamento automatizado
        """
        
        self.logger.info("Iniciando sistema de treinamento automatizado")
        
        # Configurar agendamento
        self.configurar_agendamento()
        
        # Iniciar thread de processamento
        self.thread_treinamento = threading.Thread(
            target=self.processar_fila_treinamento,
            daemon=True
        )
        self.thread_treinamento.start()
        
        self.logger.info("Sistema de treinamento automatizado iniciado")
    
    def parar_sistema_treinamento(self):
        """
        Para o sistema de treinamento
        """
        
        self.parar_sistema = True
        
        if self.thread_treinamento and self.thread_treinamento.is_alive():
            self.thread_treinamento.join(timeout=30)
        
        self.logger.info("Sistema de treinamento automatizado parado")
    
    def treinar_manual(self) -> Dict[str, Any]:
        """
        Executa treinamento manual
        """
        
        self.adicionar_treinamento_fila('manual')
        
        # Aguardar conclusão
        while self.treinamento_ativo or not self.fila_treinamento.empty():
            time.sleep(5)
        
        # Retornar último resultado
        if self.historico_treinamentos:
            return self.historico_treinamentos[-1]
        
        return {'sucesso': False, 'erro': 'Nenhum treinamento executado'}
    
    def obter_status_sistema(self) -> Dict[str, Any]:
        """
        Obtém status atual do sistema
        """
        
        return {
            'sistema_ativo': not self.parar_sistema,
            'treinamento_ativo': self.treinamento_ativo,
            'fila_tamanho': self.fila_treinamento.qsize(),
            'total_treinamentos': len(self.historico_treinamentos),
            'ultimo_treinamento': self.historico_treinamentos[-1] if self.historico_treinamentos else None,
            'configuracao': self.config
        }


def iniciar_treinamento_automatizado():
    """
    Função principal para iniciar o sistema
    """
    
    sistema = SistemaTreinamentoAutomatizado()
    sistema.iniciar_sistema()
    
    try