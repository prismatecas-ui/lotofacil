#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serviço de Métricas em Tempo Real
Sistema completo para monitoramento de performance das predições
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import statistics
from collections import defaultdict, deque

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionMetric:
    """Estrutura para armazenar métricas de uma predição"""
    prediction_id: str
    timestamp: datetime
    predicted_numbers: List[int]
    actual_numbers: Optional[List[int]] = None
    accuracy_score: Optional[float] = None
    hits_count: Optional[int] = None
    processing_time_ms: Optional[float] = None
    model_confidence: Optional[float] = None
    contest_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário para serialização"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class PerformanceStats:
    """Estatísticas de performance agregadas"""
    total_predictions: int = 0
    total_evaluated: int = 0
    average_accuracy: float = 0.0
    average_hits: float = 0.0
    average_processing_time: float = 0.0
    best_accuracy: float = 0.0
    worst_accuracy: float = 0.0
    accuracy_trend: List[float] = None
    last_24h_predictions: int = 0
    last_7d_accuracy: float = 0.0
    
    def __post_init__(self):
        if self.accuracy_trend is None:
            self.accuracy_trend = []

class MetricsDatabase:
    """Gerenciador de banco de dados para métricas"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Inicializa as tabelas do banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    predicted_numbers TEXT NOT NULL,
                    actual_numbers TEXT,
                    accuracy_score REAL,
                    hits_count INTEGER,
                    processing_time_ms REAL,
                    model_confidence REAL,
                    contest_number INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    period TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.commit()
    
    def save_prediction(self, metric: PredictionMetric):
        """Salva uma métrica de predição"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions 
                    (id, timestamp, predicted_numbers, actual_numbers, 
                     accuracy_score, hits_count, processing_time_ms, 
                     model_confidence, contest_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.prediction_id,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.predicted_numbers),
                    json.dumps(metric.actual_numbers) if metric.actual_numbers else None,
                    metric.accuracy_score,
                    metric.hits_count,
                    metric.processing_time_ms,
                    metric.model_confidence,
                    metric.contest_number
                ))
                conn.commit()
    
    def get_predictions(self, limit: int = 100, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> List[PredictionMetric]:
        """Recupera predições do banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM predictions"
            params = []
            
            if start_date or end_date:
                query += " WHERE"
                conditions = []
                
                if start_date:
                    conditions.append(" timestamp >= ?")
                    params.append(start_date.isoformat())
                
                if end_date:
                    conditions.append(" timestamp <= ?")
                    params.append(end_date.isoformat())
                
                query += " AND".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            predictions = []
            for row in rows:
                predictions.append(PredictionMetric(
                    prediction_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    predicted_numbers=json.loads(row[2]),
                    actual_numbers=json.loads(row[3]) if row[3] else None,
                    accuracy_score=row[4],
                    hits_count=row[5],
                    processing_time_ms=row[6],
                    model_confidence=row[7],
                    contest_number=row[8]
                ))
            
            return predictions
    
    def log_performance_metric(self, metric_name: str, value: float, period: str = "realtime"):
        """Registra uma métrica de performance"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_log (timestamp, metric_name, metric_value, period)
                    VALUES (?, ?, ?, ?)
                """, (datetime.now().isoformat(), metric_name, value, period))
                conn.commit()
    
    def save_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Salva um alerta no banco de dados"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts (timestamp, alert_type, message, severity)
                    VALUES (?, ?, ?, ?)
                """, (datetime.now().isoformat(), alert_type, message, severity))
                conn.commit()

class RealTimeMetricsService:
    """Serviço principal de métricas em tempo real"""
    
    def __init__(self, db_path: str = "api/metrics.db"):
        self.db = MetricsDatabase(db_path)
        self.recent_metrics = deque(maxlen=1000)  # Buffer circular para métricas recentes
        self.performance_thresholds = {
            'min_accuracy': 0.15,  # 15% mínimo de acurácia
            'max_processing_time': 5000,  # 5 segundos máximo
            'accuracy_drop_threshold': 0.05  # 5% de queda na acurácia
        }
        self.alert_callbacks = []
        self.running = False
        self.monitor_thread = None
    
    def add_alert_callback(self, callback):
        """Adiciona callback para alertas"""
        self.alert_callbacks.append(callback)
    
    def record_prediction(self, prediction_id: str, predicted_numbers: List[int],
                         processing_time_ms: float = None, 
                         model_confidence: float = None,
                         contest_number: int = None) -> PredictionMetric:
        """Registra uma nova predição"""
        metric = PredictionMetric(
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            predicted_numbers=predicted_numbers,
            processing_time_ms=processing_time_ms,
            model_confidence=model_confidence,
            contest_number=contest_number
        )
        
        self.db.save_prediction(metric)
        self.recent_metrics.append(metric)
        
        # Verifica alertas de performance
        self._check_performance_alerts(metric)
        
        logger.info(f"Predição registrada: {prediction_id}")
        return metric
    
    def update_prediction_result(self, prediction_id: str, actual_numbers: List[int]):
        """Atualiza uma predição com o resultado real"""
        # Busca a predição existente
        predictions = self.db.get_predictions(limit=1000)
        target_prediction = None
        
        for pred in predictions:
            if pred.prediction_id == prediction_id:
                target_prediction = pred
                break
        
        if not target_prediction:
            logger.warning(f"Predição não encontrada: {prediction_id}")
            return
        
        # Calcula métricas de acurácia
        hits = len(set(target_prediction.predicted_numbers) & set(actual_numbers))
        accuracy = hits / len(actual_numbers) if actual_numbers else 0
        
        # Atualiza a predição
        target_prediction.actual_numbers = actual_numbers
        target_prediction.hits_count = hits
        target_prediction.accuracy_score = accuracy
        
        self.db.save_prediction(target_prediction)
        
        # Registra métricas de performance
        self.db.log_performance_metric("accuracy", accuracy)
        self.db.log_performance_metric("hits_count", hits)
        
        logger.info(f"Resultado atualizado para {prediction_id}: {hits} acertos, {accuracy:.2%} acurácia")
    
    def get_performance_stats(self, period_hours: int = 24) -> PerformanceStats:
        """Calcula estatísticas de performance para um período"""
        start_date = datetime.now() - timedelta(hours=period_hours)
        predictions = self.db.get_predictions(start_date=start_date)
        
        if not predictions:
            return PerformanceStats()
        
        # Filtra predições avaliadas (com resultado real)
        evaluated = [p for p in predictions if p.accuracy_score is not None]
        
        stats = PerformanceStats()
        stats.total_predictions = len(predictions)
        stats.total_evaluated = len(evaluated)
        
        if evaluated:
            accuracies = [p.accuracy_score for p in evaluated]
            hits = [p.hits_count for p in evaluated]
            
            stats.average_accuracy = statistics.mean(accuracies)
            stats.average_hits = statistics.mean(hits)
            stats.best_accuracy = max(accuracies)
            stats.worst_accuracy = min(accuracies)
            stats.accuracy_trend = accuracies[-10:]  # Últimas 10 predições
        
        # Tempo de processamento
        processing_times = [p.processing_time_ms for p in predictions if p.processing_time_ms]
        if processing_times:
            stats.average_processing_time = statistics.mean(processing_times)
        
        # Métricas específicas de período
        stats.last_24h_predictions = len([p for p in predictions 
                                        if p.timestamp >= datetime.now() - timedelta(hours=24)])
        
        week_predictions = [p for p in predictions 
                          if p.timestamp >= datetime.now() - timedelta(days=7) 
                          and p.accuracy_score is not None]
        if week_predictions:
            stats.last_7d_accuracy = statistics.mean([p.accuracy_score for p in week_predictions])
        
        return stats
    
    def _check_performance_alerts(self, metric: PredictionMetric):
        """Verifica se há alertas de performance a serem disparados"""
        # Alerta de tempo de processamento
        if (metric.processing_time_ms and 
            metric.processing_time_ms > self.performance_thresholds['max_processing_time']):
            
            message = f"Tempo de processamento alto: {metric.processing_time_ms:.2f}ms"
            self.db.save_alert("high_processing_time", message, "warning")
            self._trigger_alert("high_processing_time", message)
        
        # Verifica tendência de acurácia (apenas se temos resultados)
        if len(self.recent_metrics) >= 10:
            recent_evaluated = [m for m in list(self.recent_metrics)[-10:] 
                              if m.accuracy_score is not None]
            
            if len(recent_evaluated) >= 5:
                recent_accuracies = [m.accuracy_score for m in recent_evaluated]
                avg_recent = statistics.mean(recent_accuracies)
                
                if avg_recent < self.performance_thresholds['min_accuracy']:
                    message = f"Acurácia baixa detectada: {avg_recent:.2%}"
                    self.db.save_alert("low_accuracy", message, "critical")
                    self._trigger_alert("low_accuracy", message)
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Dispara callbacks de alerta"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message)
            except Exception as e:
                logger.error(f"Erro ao executar callback de alerta: {e}")
    
    def start_monitoring(self):
        """Inicia o monitoramento em background"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoramento de métricas iniciado")
    
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoramento de métricas parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                # Calcula e registra métricas agregadas a cada 5 minutos
                stats = self.get_performance_stats(period_hours=1)
                
                if stats.total_evaluated > 0:
                    self.db.log_performance_metric("avg_accuracy_1h", stats.average_accuracy, "hourly")
                    self.db.log_performance_metric("avg_hits_1h", stats.average_hits, "hourly")
                
                time.sleep(300)  # 5 minutos
                
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(60)  # Espera 1 minuto em caso de erro
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Recupera alertas recentes"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, alert_type, message, severity, resolved
                FROM alerts 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, ((datetime.now() - timedelta(hours=hours)).isoformat(),))
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'timestamp': row[0],
                    'alert_type': row[1],
                    'message': row[2],
                    'severity': row[3],
                    'resolved': bool(row[4])
                })
            
            return alerts
    
    def export_metrics(self, start_date: datetime, end_date: datetime, 
                      format: str = "json") -> str:
        """Exporta métricas para análise externa"""
        predictions = self.db.get_predictions(start_date=start_date, end_date=end_date)
        
        if format == "json":
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'predictions': [p.to_dict() for p in predictions],
                'summary': asdict(self.get_performance_stats())
            }
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Formato não suportado: {format}")

# Instância global do serviço
metrics_service = RealTimeMetricsService()

# Função de conveniência para uso externo
def record_prediction(prediction_id: str, predicted_numbers: List[int], 
                     processing_time_ms: float = None, 
                     model_confidence: float = None,
                     contest_number: int = None) -> PredictionMetric:
    """Função de conveniência para registrar predições"""
    return metrics_service.record_prediction(
        prediction_id, predicted_numbers, processing_time_ms, 
        model_confidence, contest_number
    )

def update_prediction_result(prediction_id: str, actual_numbers: List[int]):
    """Função de conveniência para atualizar resultados"""
    metrics_service.update_prediction_result(prediction_id, actual_numbers)

def get_performance_stats(period_hours: int = 24) -> PerformanceStats:
    """Função de conveniência para obter estatísticas"""
    return metrics_service.get_performance_stats(period_hours)

if __name__ == "__main__":
    # Teste básico do serviço
    print("Testando serviço de métricas...")
    
    # Inicia monitoramento
    metrics_service.start_monitoring()
    
    # Simula algumas predições
    pred1 = record_prediction("test_001", [1, 2, 3, 4, 5], 1500.0, 0.85)
    pred2 = record_prediction("test_002", [6, 7, 8, 9, 10], 1200.0, 0.92)
    
    # Simula resultados
    update_prediction_result("test_001", [1, 2, 15, 16, 17])  # 2 acertos
    update_prediction_result("test_002", [6, 7, 8, 18, 19])   # 3 acertos
    
    # Mostra estatísticas
    stats = get_performance_stats()
    print(f"Estatísticas: {stats.total_predictions} predições, {stats.average_accuracy:.2%} acurácia média")
    
    # Para monitoramento
    metrics_service.stop_monitoring()
    print("Teste concluído!")