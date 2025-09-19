#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Tracking de Acurácia
Monitora automaticamente a precisão das predições comparando com resultados reais
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import logging
import statistics
from collections import defaultdict

# Importa o serviço de métricas
from .metrics_service import metrics_service, PredictionMetric
from .caixa_api import CaixaAPI

logger = logging.getLogger(__name__)

@dataclass
class AccuracyAnalysis:
    """Análise detalhada de acurácia"""
    prediction_id: str
    contest_number: int
    predicted_numbers: List[int]
    actual_numbers: List[int]
    hits_count: int
    accuracy_percentage: float
    hit_positions: List[int]  # Posições dos números que acertaram
    miss_numbers: List[int]   # Números preditos que não saíram
    surprise_numbers: List[int]  # Números que saíram mas não foram preditos
    confidence_score: Optional[float] = None
    analysis_timestamp: datetime = None
    
    def __post_init__(self):
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now()

@dataclass
class AccuracyTrend:
    """Tendência de acurácia ao longo do tempo"""
    period_start: datetime
    period_end: datetime
    total_predictions: int
    average_accuracy: float
    accuracy_variance: float
    best_prediction: Optional[AccuracyAnalysis]
    worst_prediction: Optional[AccuracyAnalysis]
    trend_direction: str  # 'improving', 'declining', 'stable'
    confidence_correlation: float  # Correlação entre confiança e acurácia

class AccuracyTracker:
    """Sistema principal de tracking de acurácia"""
    
    def __init__(self, db_path: str = "api/accuracy_tracking.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.caixa_api = CaixaAPI()
        self.auto_check_enabled = True
        self.check_interval = 3600  # Verifica a cada hora
        self.monitoring_thread = None
        self.running = False
        self.callbacks = []
        
        self._init_database()
        
    def _init_database(self):
        """Inicializa banco de dados específico para tracking de acurácia"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE NOT NULL,
                    contest_number INTEGER NOT NULL,
                    predicted_numbers TEXT NOT NULL,
                    actual_numbers TEXT NOT NULL,
                    hits_count INTEGER NOT NULL,
                    accuracy_percentage REAL NOT NULL,
                    hit_positions TEXT NOT NULL,
                    miss_numbers TEXT NOT NULL,
                    surprise_numbers TEXT NOT NULL,
                    confidence_score REAL,
                    analysis_timestamp TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    average_accuracy REAL NOT NULL,
                    accuracy_variance REAL NOT NULL,
                    trend_direction TEXT NOT NULL,
                    confidence_correlation REAL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    contest_number INTEGER NOT NULL,
                    expected_draw_date TEXT NOT NULL,
                    checked BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    def add_callback(self, callback: Callable[[AccuracyAnalysis], None]):
        """Adiciona callback para ser executado quando uma análise é completada"""
        self.callbacks.append(callback)
    
    def schedule_accuracy_check(self, prediction_id: str, contest_number: int, 
                               expected_draw_date: datetime):
        """Agenda verificação de acurácia para uma predição"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pending_checks 
                    (prediction_id, contest_number, expected_draw_date, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    prediction_id,
                    contest_number,
                    expected_draw_date.isoformat(),
                    datetime.now().isoformat()
                ))
                conn.commit()
        
        logger.info(f"Verificação agendada para predição {prediction_id} no concurso {contest_number}")
    
    def check_prediction_accuracy(self, prediction_id: str, 
                                 actual_numbers: List[int]) -> Optional[AccuracyAnalysis]:
        """Verifica a acurácia de uma predição específica"""
        # Busca a predição no serviço de métricas
        predictions = metrics_service.db.get_predictions(limit=1000)
        target_prediction = None
        
        for pred in predictions:
            if pred.prediction_id == prediction_id:
                target_prediction = pred
                break
        
        if not target_prediction:
            logger.warning(f"Predição não encontrada: {prediction_id}")
            return None
        
        # Calcula análise detalhada
        predicted_set = set(target_prediction.predicted_numbers)
        actual_set = set(actual_numbers)
        
        hits = predicted_set & actual_set
        hits_count = len(hits)
        accuracy_percentage = (hits_count / len(actual_numbers)) * 100
        
        # Identifica posições dos acertos
        hit_positions = []
        for i, num in enumerate(target_prediction.predicted_numbers):
            if num in actual_set:
                hit_positions.append(i)
        
        miss_numbers = list(predicted_set - actual_set)
        surprise_numbers = list(actual_set - predicted_set)
        
        # Cria análise
        analysis = AccuracyAnalysis(
            prediction_id=prediction_id,
            contest_number=target_prediction.contest_number or 0,
            predicted_numbers=target_prediction.predicted_numbers,
            actual_numbers=actual_numbers,
            hits_count=hits_count,
            accuracy_percentage=accuracy_percentage,
            hit_positions=hit_positions,
            miss_numbers=miss_numbers,
            surprise_numbers=surprise_numbers,
            confidence_score=target_prediction.model_confidence
        )
        
        # Salva análise no banco
        self._save_analysis(analysis)
        
        # Atualiza métricas no serviço principal
        metrics_service.update_prediction_result(prediction_id, actual_numbers)
        
        # Executa callbacks
        for callback in self.callbacks:
            try:
                callback(analysis)
            except Exception as e:
                logger.error(f"Erro ao executar callback: {e}")
        
        logger.info(f"Análise concluída: {prediction_id} - {hits_count} acertos ({accuracy_percentage:.1f}%)")
        return analysis
    
    def _save_analysis(self, analysis: AccuracyAnalysis):
        """Salva análise de acurácia no banco de dados"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO accuracy_analysis 
                    (prediction_id, contest_number, predicted_numbers, actual_numbers,
                     hits_count, accuracy_percentage, hit_positions, miss_numbers,
                     surprise_numbers, confidence_score, analysis_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis.prediction_id,
                    analysis.contest_number,
                    json.dumps(analysis.predicted_numbers),
                    json.dumps(analysis.actual_numbers),
                    analysis.hits_count,
                    analysis.accuracy_percentage,
                    json.dumps(analysis.hit_positions),
                    json.dumps(analysis.miss_numbers),
                    json.dumps(analysis.surprise_numbers),
                    analysis.confidence_score,
                    analysis.analysis_timestamp.isoformat()
                ))
                conn.commit()
    
    def get_accuracy_history(self, limit: int = 100, 
                           start_date: Optional[datetime] = None) -> List[AccuracyAnalysis]:
        """Recupera histórico de análises de acurácia"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM accuracy_analysis"
            params = []
            
            if start_date:
                query += " WHERE analysis_timestamp >= ?"
                params.append(start_date.isoformat())
            
            query += " ORDER BY analysis_timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            analyses = []
            for row in rows:
                analyses.append(AccuracyAnalysis(
                    prediction_id=row[1],
                    contest_number=row[2],
                    predicted_numbers=json.loads(row[3]),
                    actual_numbers=json.loads(row[4]),
                    hits_count=row[5],
                    accuracy_percentage=row[6],
                    hit_positions=json.loads(row[7]),
                    miss_numbers=json.loads(row[8]),
                    surprise_numbers=json.loads(row[9]),
                    confidence_score=row[10],
                    analysis_timestamp=datetime.fromisoformat(row[11])
                ))
            
            return analyses
    
    def calculate_accuracy_trend(self, days: int = 30) -> AccuracyTrend:
        """Calcula tendência de acurácia para um período"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        analyses = self.get_accuracy_history(start_date=start_date)
        
        if not analyses:
            return AccuracyTrend(
                period_start=start_date,
                period_end=end_date,
                total_predictions=0,
                average_accuracy=0.0,
                accuracy_variance=0.0,
                best_prediction=None,
                worst_prediction=None,
                trend_direction='stable',
                confidence_correlation=0.0
            )
        
        # Calcula estatísticas
        accuracies = [a.accuracy_percentage for a in analyses]
        average_accuracy = statistics.mean(accuracies)
        accuracy_variance = statistics.variance(accuracies) if len(accuracies) > 1 else 0.0
        
        # Encontra melhor e pior predição
        best_prediction = max(analyses, key=lambda x: x.accuracy_percentage)
        worst_prediction = min(analyses, key=lambda x: x.accuracy_percentage)
        
        # Calcula tendência
        trend_direction = self._calculate_trend_direction(analyses)
        
        # Correlação entre confiança e acurácia
        confidence_correlation = self._calculate_confidence_correlation(analyses)
        
        trend = AccuracyTrend(
            period_start=start_date,
            period_end=end_date,
            total_predictions=len(analyses),
            average_accuracy=average_accuracy,
            accuracy_variance=accuracy_variance,
            best_prediction=best_prediction,
            worst_prediction=worst_prediction,
            trend_direction=trend_direction,
            confidence_correlation=confidence_correlation
        )
        
        # Salva tendência no banco
        self._save_trend(trend)
        
        return trend
    
    def _calculate_trend_direction(self, analyses: List[AccuracyAnalysis]) -> str:
        """Calcula direção da tendência de acurácia"""
        if len(analyses) < 5:
            return 'stable'
        
        # Divide em duas metades e compara
        mid_point = len(analyses) // 2
        recent_half = analyses[:mid_point]  # Mais recentes primeiro
        older_half = analyses[mid_point:]
        
        recent_avg = statistics.mean([a.accuracy_percentage for a in recent_half])
        older_avg = statistics.mean([a.accuracy_percentage for a in older_half])
        
        difference = recent_avg - older_avg
        
        if difference > 2.0:  # Melhoria de mais de 2%
            return 'improving'
        elif difference < -2.0:  # Piora de mais de 2%
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_confidence_correlation(self, analyses: List[AccuracyAnalysis]) -> float:
        """Calcula correlação entre confiança do modelo e acurácia real"""
        valid_analyses = [a for a in analyses if a.confidence_score is not None]
        
        if len(valid_analyses) < 3:
            return 0.0
        
        confidences = [a.confidence_score for a in valid_analyses]
        accuracies = [a.accuracy_percentage for a in valid_analyses]
        
        try:
            # Correlação de Pearson simples
            n = len(confidences)
            sum_conf = sum(confidences)
            sum_acc = sum(accuracies)
            sum_conf_sq = sum(c * c for c in confidences)
            sum_acc_sq = sum(a * a for a in accuracies)
            sum_conf_acc = sum(c * a for c, a in zip(confidences, accuracies))
            
            numerator = n * sum_conf_acc - sum_conf * sum_acc
            denominator = ((n * sum_conf_sq - sum_conf * sum_conf) * 
                          (n * sum_acc_sq - sum_acc * sum_acc)) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))  # Limita entre -1 e 1
            
        except Exception as e:
            logger.error(f"Erro ao calcular correlação: {e}")
            return 0.0
    
    def _save_trend(self, trend: AccuracyTrend):
        """Salva tendência no banco de dados"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO accuracy_trends 
                    (period_start, period_end, total_predictions, average_accuracy,
                     accuracy_variance, trend_direction, confidence_correlation, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trend.period_start.isoformat(),
                    trend.period_end.isoformat(),
                    trend.total_predictions,
                    trend.average_accuracy,
                    trend.accuracy_variance,
                    trend.trend_direction,
                    trend.confidence_correlation,
                    datetime.now().isoformat()
                ))
                conn.commit()
    
    def start_auto_checking(self):
        """Inicia verificação automática de resultados"""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._auto_check_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Verificação automática de acurácia iniciada")
    
    def stop_auto_checking(self):
        """Para verificação automática"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Verificação automática de acurácia parada")
    
    def _auto_check_loop(self):
        """Loop principal de verificação automática"""
        while self.running:
            try:
                self._process_pending_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Erro no loop de verificação automática: {e}")
                time.sleep(300)  # Espera 5 minutos em caso de erro
    
    def _process_pending_checks(self):
        """Processa verificações pendentes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, prediction_id, contest_number, expected_draw_date
                FROM pending_checks 
                WHERE checked = FALSE AND expected_draw_date <= ?
                ORDER BY expected_draw_date
            """, (datetime.now().isoformat(),))
            
            pending = cursor.fetchall()
        
        for check_id, prediction_id, contest_number, expected_date in pending:
            try:
                # Busca resultado do concurso
                result = self.caixa_api.get_contest_result(contest_number)
                
                if result and 'dezenas' in result:
                    actual_numbers = [int(num) for num in result['dezenas']]
                    
                    # Verifica acurácia
                    analysis = self.check_prediction_accuracy(prediction_id, actual_numbers)
                    
                    if analysis:
                        # Marca como verificado
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute(
                                "UPDATE pending_checks SET checked = TRUE WHERE id = ?",
                                (check_id,)
                            )
                            conn.commit()
                        
                        logger.info(f"Verificação automática concluída: {prediction_id}")
                
            except Exception as e:
                logger.error(f"Erro ao verificar predição {prediction_id}: {e}")
    
    def get_accuracy_summary(self, days: int = 30) -> Dict[str, Any]:
        """Gera resumo completo de acurácia"""
        trend = self.calculate_accuracy_trend(days)
        recent_analyses = self.get_accuracy_history(limit=50)
        
        # Estatísticas por número de acertos
        hits_distribution = defaultdict(int)
        for analysis in recent_analyses:
            hits_distribution[analysis.hits_count] += 1
        
        # Números mais e menos acertados
        number_hits = defaultdict(int)
        number_misses = defaultdict(int)
        
        for analysis in recent_analyses:
            for num in analysis.predicted_numbers:
                if num in analysis.actual_numbers:
                    number_hits[num] += 1
                else:
                    number_misses[num] += 1
        
        most_accurate_numbers = sorted(number_hits.items(), key=lambda x: x[1], reverse=True)[:10]
        least_accurate_numbers = sorted(number_misses.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'period_summary': {
                'days': days,
                'total_predictions': trend.total_predictions,
                'average_accuracy': trend.average_accuracy,
                'trend_direction': trend.trend_direction,
                'confidence_correlation': trend.confidence_correlation
            },
            'hits_distribution': dict(hits_distribution),
            'best_prediction': {
                'id': trend.best_prediction.prediction_id if trend.best_prediction else None,
                'accuracy': trend.best_prediction.accuracy_percentage if trend.best_prediction else 0,
                'hits': trend.best_prediction.hits_count if trend.best_prediction else 0
            },
            'worst_prediction': {
                'id': trend.worst_prediction.prediction_id if trend.worst_prediction else None,
                'accuracy': trend.worst_prediction.accuracy_percentage if trend.worst_prediction else 0,
                'hits': trend.worst_prediction.hits_count if trend.worst_prediction else 0
            },
            'number_performance': {
                'most_accurate': most_accurate_numbers,
                'least_accurate': least_accurate_numbers
            },
            'recent_analyses': [{
                'prediction_id': a.prediction_id,
                'contest_number': a.contest_number,
                'hits_count': a.hits_count,
                'accuracy_percentage': a.accuracy_percentage,
                'timestamp': a.analysis_timestamp.isoformat()
            } for a in recent_analyses[:10]]
        }

# Instância global do tracker
accuracy_tracker = AccuracyTracker()

# Funções de conveniência
def schedule_check(prediction_id: str, contest_number: int, expected_draw_date: datetime):
    """Agenda verificação de acurácia"""
    accuracy_tracker.schedule_accuracy_check(prediction_id, contest_number, expected_draw_date)

def check_accuracy(prediction_id: str, actual_numbers: List[int]) -> Optional[AccuracyAnalysis]:
    """Verifica acurácia de uma predição"""
    return accuracy_tracker.check_prediction_accuracy(prediction_id, actual_numbers)

def get_accuracy_summary(days: int = 30) -> Dict[str, Any]:
    """Obtém resumo de acurácia"""
    return accuracy_tracker.get_accuracy_summary(days)

if __name__ == "__main__":
    # Teste do sistema
    print("Testando sistema de tracking de acurácia...")
    
    # Inicia verificação automática
    accuracy_tracker.start_auto_checking()
    
    # Simula verificação manual
    analysis = check_accuracy("test_001", [1, 2, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    if analysis:
        print(f"Análise: {analysis.hits_count} acertos, {analysis.accuracy_percentage:.1f}% acurácia")
    
    # Mostra resumo
    summary = get_accuracy_summary(7)
    print(f"Resumo: {summary['period_summary']['total_predictions']} predições analisadas")
    
    # Para verificação automática
    accuracy_tracker.stop_auto_checking()
    print("Teste concluído!")