#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Logging Avançado para Experimentos de Otimização da IA Lotofácil

Este módulo fornece funcionalidades completas de logging para rastrear:
- Experimentos de treinamento de modelos
- Métricas de performance
- Hiperparâmetros utilizados
- Resultados e comparações
- Análises exploratórias
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path

class ExperimentLogger:
    """
    Logger especializado para experimentos de otimização da IA
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experimentos/logs"):
        """
        Inicializa o logger de experimentos
        
        Args:
            experiment_name: Nome do experimento
            base_dir: Diretório base para logs
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp único para este experimento
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        # Arquivos de log
        self.log_file = self.base_dir / f"{self.experiment_id}.log"
        self.metrics_file = self.base_dir / f"{self.experiment_id}_metrics.json"
        self.config_file = self.base_dir / f"{self.experiment_id}_config.json"
        
        # Configurar logger
        self._setup_logger()
        
        # Dados do experimento
        self.experiment_data = {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "metrics": {},
            "results": {},
            "notes": []
        }
        
        self.logger.info(f"Iniciando experimento: {self.experiment_id}")
    
    def _setup_logger(self):
        """
        Configura o sistema de logging
        """
        # Criar logger específico para este experimento
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.DEBUG)
        
        # Remover handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Handler para arquivo
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formato das mensagens
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Adicionar handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Registra configuração do experimento
        
        Args:
            config: Dicionário com configurações
        """
        self.experiment_data["config"].update(config)
        self.logger.info(f"Configuração registrada: {json.dumps(config, indent=2)}")
        self._save_config()
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Registra hiperparâmetros do modelo
        
        Args:
            hyperparams: Dicionário com hiperparâmetros
        """
        self.experiment_data["config"]["hyperparameters"] = hyperparams
        self.logger.info(f"Hiperparâmetros: {json.dumps(hyperparams, indent=2)}")
        self._save_config()
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """
        Registra informações sobre os dados utilizados
        
        Args:
            data_info: Informações sobre dataset
        """
        self.experiment_data["config"]["data_info"] = data_info
        self.logger.info(f"Informações dos dados: {json.dumps(data_info, indent=2)}")
        self._save_config()
    
    def log_metric(self, metric_name: str, value: float, epoch: Optional[int] = None):
        """
        Registra uma métrica
        
        Args:
            metric_name: Nome da métrica
            value: Valor da métrica
            epoch: Época (opcional)
        """
        if metric_name not in self.experiment_data["metrics"]:
            self.experiment_data["metrics"][metric_name] = []
        
        metric_entry = {
            "value": float(value),
            "timestamp": datetime.now().isoformat()
        }
        
        if epoch is not None:
            metric_entry["epoch"] = epoch
        
        self.experiment_data["metrics"][metric_name].append(metric_entry)
        
        log_msg = f"Métrica {metric_name}: {value}"
        if epoch is not None:
            log_msg += f" (época {epoch})"
        
        self.logger.info(log_msg)
        self._save_metrics()
    
    def log_metrics_batch(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """
        Registra múltiplas métricas de uma vez
        
        Args:
            metrics: Dicionário com métricas
            epoch: Época (opcional)
        """
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, epoch)
    
    def log_model_architecture(self, model_summary: str):
        """
        Registra arquitetura do modelo
        
        Args:
            model_summary: Resumo da arquitetura do modelo
        """
        self.experiment_data["config"]["model_architecture"] = model_summary
        self.logger.info(f"Arquitetura do modelo registrada")
        self._save_config()
    
    def log_training_progress(self, epoch: int, loss: float, metrics: Dict[str, float]):
        """
        Registra progresso do treinamento
        
        Args:
            epoch: Número da época
            loss: Valor da loss
            metrics: Métricas adicionais
        """
        self.log_metric("loss", loss, epoch)
        self.log_metrics_batch(metrics, epoch)
        
        progress_msg = f"Época {epoch}: Loss={loss:.4f}"
        for name, value in metrics.items():
            progress_msg += f", {name}={value:.4f}"
        
        self.logger.info(progress_msg)
    
    def log_prediction_results(self, predictions: List[List[int]], 
                             probabilities: List[float], 
                             actual_results: Optional[List[List[int]]] = None):
        """
        Registra resultados de predições
        
        Args:
            predictions: Lista de predições (jogos)
            probabilities: Probabilidades associadas
            actual_results: Resultados reais (opcional)
        """
        results_data = {
            "predictions": predictions,
            "probabilities": probabilities,
            "timestamp": datetime.now().isoformat()
        }
        
        if actual_results is not None:
            results_data["actual_results"] = actual_results
            # Calcular acertos
            acertos = self._calculate_hits(predictions, actual_results)
            results_data["hits"] = acertos
            self.logger.info(f"Predições com acertos: {acertos}")
        
        if "predictions" not in self.experiment_data["results"]:
            self.experiment_data["results"]["predictions"] = []
        
        self.experiment_data["results"]["predictions"].append(results_data)
        self.logger.info(f"Registradas {len(predictions)} predições")
        self._save_metrics()
    
    def _calculate_hits(self, predictions: List[List[int]], 
                       actual_results: List[List[int]]) -> List[int]:
        """
        Calcula número de acertos para cada predição
        
        Args:
            predictions: Predições feitas
            actual_results: Resultados reais
            
        Returns:
            Lista com número de acertos para cada predição
        """
        hits = []
        for pred, actual in zip(predictions, actual_results):
            hit_count = len(set(pred) & set(actual))
            hits.append(hit_count)
        return hits
    
    def log_note(self, note: str):
        """
        Adiciona uma nota ao experimento
        
        Args:
            note: Texto da nota
        """
        note_entry = {
            "note": note,
            "timestamp": datetime.now().isoformat()
        }
        self.experiment_data["notes"].append(note_entry)
        self.logger.info(f"Nota: {note}")
        self._save_metrics()
    
    def log_error(self, error: Exception, context: str = ""):
        """
        Registra um erro ocorrido durante o experimento
        
        Args:
            error: Exceção ocorrida
            context: Contexto adicional
        """
        error_msg = f"Erro {context}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        error_entry = {
            "error": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        if "errors" not in self.experiment_data:
            self.experiment_data["errors"] = []
        
        self.experiment_data["errors"].append(error_entry)
        self._save_metrics()
    
    def finish_experiment(self, final_results: Optional[Dict[str, Any]] = None):
        """
        Finaliza o experimento
        
        Args:
            final_results: Resultados finais (opcional)
        """
        self.experiment_data["end_time"] = datetime.now().isoformat()
        
        if final_results:
            self.experiment_data["results"].update(final_results)
        
        # Calcular duração
        start_time = datetime.fromisoformat(self.experiment_data["start_time"])
        end_time = datetime.fromisoformat(self.experiment_data["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.experiment_data["duration_seconds"] = duration
        
        self.logger.info(f"Experimento finalizado. Duração: {duration:.2f} segundos")
        self._save_metrics()
        
        # Gerar resumo
        self._generate_summary()
    
    def _save_config(self):
        """
        Salva configuração em arquivo JSON
        """
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data["config"], f, indent=2, ensure_ascii=False)
    
    def _save_metrics(self):
        """
        Salva métricas e resultados em arquivo JSON
        """
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)
    
    def _generate_summary(self):
        """
        Gera resumo do experimento
        """
        summary_file = self.base_dir / f"{self.experiment_id}_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RESUMO DO EXPERIMENTO: {self.experiment_name}\n")
            f.write(f"ID: {self.experiment_id}\n")
            f.write(f"Início: {self.experiment_data['start_time']}\n")
            f.write(f"Fim: {self.experiment_data.get('end_time', 'N/A')}\n")
            f.write(f"Duração: {self.experiment_data.get('duration_seconds', 0):.2f}s\n\n")
            
            # Configurações
            if self.experiment_data["config"]:
                f.write("CONFIGURAÇÕES:\n")
                for key, value in self.experiment_data["config"].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Métricas finais
            if self.experiment_data["metrics"]:
                f.write("MÉTRICAS FINAIS:\n")
                for metric_name, values in self.experiment_data["metrics"].items():
                    if values:
                        final_value = values[-1]["value"]
                        f.write(f"  {metric_name}: {final_value:.4f}\n")
                f.write("\n")
            
            # Notas
            if self.experiment_data["notes"]:
                f.write("NOTAS:\n")
                for note in self.experiment_data["notes"]:
                    f.write(f"  - {note['note']} ({note['timestamp']})\n")
        
        self.logger.info(f"Resumo salvo em: {summary_file}")


class ExperimentManager:
    """
    Gerenciador de múltiplos experimentos
    """
    
    def __init__(self, base_dir: str = "experimentos/logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def list_experiments(self) -> List[str]:
        """
        Lista todos os experimentos realizados
        
        Returns:
            Lista com IDs dos experimentos
        """
        experiments = []
        for file in self.base_dir.glob("*_metrics.json"):
            experiment_id = file.stem.replace("_metrics", "")
            experiments.append(experiment_id)
        return sorted(experiments)
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Carrega dados de um experimento
        
        Args:
            experiment_id: ID do experimento
            
        Returns:
            Dados do experimento
        """
        metrics_file = self.base_dir / f"{experiment_id}_metrics.json"
        
        if not metrics_file.exists():
            raise FileNotFoundError(f"Experimento {experiment_id} não encontrado")
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_experiments(self, experiment_ids: List[str], 
                          metric_name: str) -> pd.DataFrame:
        """
        Compara métricas entre experimentos
        
        Args:
            experiment_ids: Lista de IDs dos experimentos
            metric_name: Nome da métrica para comparar
            
        Returns:
            DataFrame com comparação
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            try:
                exp_data = self.load_experiment(exp_id)
                if metric_name in exp_data.get("metrics", {}):
                    values = exp_data["metrics"][metric_name]
                    if values:
                        final_value = values[-1]["value"]
                        comparison_data.append({
                            "experiment_id": exp_id,
                            "experiment_name": exp_data.get("experiment_name", "N/A"),
                            metric_name: final_value,
                            "duration": exp_data.get("duration_seconds", 0)
                        })
            except Exception as e:
                print(f"Erro ao carregar experimento {exp_id}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def get_best_experiment(self, metric_name: str, 
                           maximize: bool = True) -> Optional[str]:
        """
        Encontra o melhor experimento baseado em uma métrica
        
        Args:
            metric_name: Nome da métrica
            maximize: Se True, busca maior valor; se False, menor valor
            
        Returns:
            ID do melhor experimento
        """
        experiments = self.list_experiments()
        
        if not experiments:
            return None
        
        comparison_df = self.compare_experiments(experiments, metric_name)
        
        if comparison_df.empty:
            return None
        
        if maximize:
            best_idx = comparison_df[metric_name].idxmax()
        else:
            best_idx = comparison_df[metric_name].idxmin()
        
        return comparison_df.loc[best_idx, "experiment_id"]


# Exemplo de uso
if __name__ == "__main__":
    # Criar logger para experimento
    logger = ExperimentLogger("teste_otimizacao")
    
    # Registrar configurações
    logger.log_config({
        "modelo": "neural_network",
        "dataset": "lotofacil_historico"
    })
    
    # Registrar hiperparâmetros
    logger.log_hyperparameters({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    })
    
    # Simular treinamento
    for epoch in range(5):
        loss = 0.5 - (epoch * 0.05)
        accuracy = 0.6 + (epoch * 0.02)
        
        logger.log_training_progress(epoch, loss, {
            "accuracy": accuracy,
            "val_accuracy": accuracy - 0.05
        })
    
    # Adicionar nota
    logger.log_note("Experimento de teste concluído com sucesso")
    
    # Finalizar experimento
    logger.finish_experiment({
        "final_accuracy": 0.68,
        "best_model_saved": True
    })
    
    print(f"Experimento salvo: {logger.experiment_id}")