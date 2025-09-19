#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integração do Sistema de Retreinamento com Métricas de Performance
Conecta o monitoramento de performance com o sistema de retreinamento automático
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import statistics
from collections import defaultdict

# Importa sistemas existentes
from .metrics_service import metrics_service, PredictionMetric
from .accuracy_tracker import accuracy_tracker, AccuracyAnalysis, AccuracyTrend

# Importa sistema de retreinamento
import sys
sys.path.append(str(Path(__file__).parent.parent))
from modelo.treinamento_automatizado import SistemaTreinamentoAutomatizado

logger = logging.getLogger(__name__)

@dataclass
class PerformanceThreshold:
    """Limites de performance para trigger de retreinamento"""
    min_accuracy: float = 15.0  # Acurácia mínima (% de acertos)
    max_accuracy_drop: float = 5.0  # Queda máxima permitida na acurácia
    min_confidence_correlation: float = 0.3  # Correlação mínima confiança-acurácia
    evaluation_period_days: int = 7  # Período de avaliação em dias
    min_predictions_for_evaluation: int = 5  # Mínimo de predições para avaliar
    consecutive_poor_predictions: int = 3  # Predições ruins consecutivas

@dataclass
class RetrainingTrigger:
    """Evento que dispara retreinamento"""
    trigger_type: str  # 'accuracy_drop', 'poor_correlation', 'consecutive_failures'
    trigger_value: float
    threshold_value: float
    description: str
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    auto_retrain: bool = True

@dataclass
class RetrainingResult:
    """Resultado de um retreinamento"""
    trigger_id: str
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    models_retrained: List[str]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    improvement_metrics: Dict[str, float]
    error_message: Optional[str] = None

class RetrainingIntegration:
    """Sistema de integração entre métricas e retreinamento"""
    
    def __init__(self, config_path: str = "api/retraining_config.json"):
        self.config_path = config_path
        self.thresholds = PerformanceThreshold()
        self.training_system = None
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # Histórico de triggers e retreinamentos
        self.trigger_history: List[RetrainingTrigger] = []
        self.retraining_history: List[RetrainingResult] = []
        
        # Callbacks para eventos
        self.trigger_callbacks: List[Callable[[RetrainingTrigger], None]] = []
        self.retraining_callbacks: List[Callable[[RetrainingResult], None]] = []
        
        self._load_config()
        self._init_training_system()
        
        # Registra callbacks no accuracy tracker
        accuracy_tracker.add_callback(self._on_accuracy_analysis)
        
    def _load_config(self):
        """Carrega configuração de retreinamento"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # Atualiza thresholds com valores do config
                threshold_config = config.get('thresholds', {})
                for key, value in threshold_config.items():
                    if hasattr(self.thresholds, key):
                        setattr(self.thresholds, key, value)
                        
                logger.info(f"Configuração carregada: {self.config_path}")
            else:
                # Cria configuração padrão
                self._save_default_config()
                
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            self._save_default_config()
    
    def _save_default_config(self):
        """Salva configuração padrão"""
        default_config = {
            'thresholds': asdict(self.thresholds),
            'monitoring': {
                'check_interval_minutes': 30,
                'auto_retrain_enabled': True,
                'notification_enabled': True
            },
            'retraining': {
                'max_concurrent_retrainings': 1,
                'cooldown_hours': 6,  # Tempo mínimo entre retreinamentos
                'backup_models_before_retrain': True
            }
        }
        
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuração padrão salva: {self.config_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar configuração padrão: {e}")
    
    def _init_training_system(self):
        """Inicializa sistema de treinamento"""
        try:
            self.training_system = SistemaTreinamentoAutomatizado()
            logger.info("Sistema de treinamento inicializado")
        except Exception as e:
            logger.error(f"Erro ao inicializar sistema de treinamento: {e}")
    
    def add_trigger_callback(self, callback: Callable[[RetrainingTrigger], None]):
        """Adiciona callback para triggers de retreinamento"""
        self.trigger_callbacks.append(callback)
    
    def add_retraining_callback(self, callback: Callable[[RetrainingResult], None]):
        """Adiciona callback para resultados de retreinamento"""
        self.retraining_callbacks.append(callback)
    
    def _on_accuracy_analysis(self, analysis: AccuracyAnalysis):
        """Callback executado quando uma análise de acurácia é completada"""
        try:
            # Verifica se a análise indica necessidade de retreinamento
            triggers = self._evaluate_retraining_need(analysis)
            
            for trigger in triggers:
                self._handle_retraining_trigger(trigger)
                
        except Exception as e:
            logger.error(f"Erro ao processar análise de acurácia: {e}")
    
    def _evaluate_retraining_need(self, analysis: AccuracyAnalysis) -> List[RetrainingTrigger]:
        """Avalia se uma análise indica necessidade de retreinamento"""
        triggers = []
        
        # 1. Verifica acurácia muito baixa
        if analysis.accuracy_percentage < self.thresholds.min_accuracy:
            triggers.append(RetrainingTrigger(
                trigger_type='low_accuracy',
                trigger_value=analysis.accuracy_percentage,
                threshold_value=self.thresholds.min_accuracy,
                description=f"Acurácia muito baixa: {analysis.accuracy_percentage:.1f}% < {self.thresholds.min_accuracy}%",
                timestamp=datetime.now(),
                severity='high' if analysis.accuracy_percentage < self.thresholds.min_accuracy * 0.7 else 'medium'
            ))
        
        # 2. Verifica queda de performance
        accuracy_trend = accuracy_tracker.calculate_accuracy_trend(self.thresholds.evaluation_period_days)
        
        if accuracy_trend.total_predictions >= self.thresholds.min_predictions_for_evaluation:
            # Compara com período anterior
            previous_trend = accuracy_tracker.calculate_accuracy_trend(
                self.thresholds.evaluation_period_days * 2
            )
            
            if previous_trend.total_predictions > 0:
                accuracy_drop = previous_trend.average_accuracy - accuracy_trend.average_accuracy
                
                if accuracy_drop > self.thresholds.max_accuracy_drop:
                    triggers.append(RetrainingTrigger(
                        trigger_type='accuracy_drop',
                        trigger_value=accuracy_drop,
                        threshold_value=self.thresholds.max_accuracy_drop,
                        description=f"Queda significativa na acurácia: {accuracy_drop:.1f}% nos últimos {self.thresholds.evaluation_period_days} dias",
                        timestamp=datetime.now(),
                        severity='high' if accuracy_drop > self.thresholds.max_accuracy_drop * 1.5 else 'medium'
                    ))
        
        # 3. Verifica correlação confiança-acurácia
        if (accuracy_trend.confidence_correlation is not None and 
            accuracy_trend.confidence_correlation < self.thresholds.min_confidence_correlation):
            
            triggers.append(RetrainingTrigger(
                trigger_type='poor_correlation',
                trigger_value=accuracy_trend.confidence_correlation,
                threshold_value=self.thresholds.min_confidence_correlation,
                description=f"Baixa correlação confiança-acurácia: {accuracy_trend.confidence_correlation:.2f}",
                timestamp=datetime.now(),
                severity='medium'
            ))
        
        # 4. Verifica predições ruins consecutivas
        recent_analyses = accuracy_tracker.get_accuracy_history(limit=self.thresholds.consecutive_poor_predictions)
        
        if len(recent_analyses) >= self.thresholds.consecutive_poor_predictions:
            consecutive_poor = all(
                a.accuracy_percentage < self.thresholds.min_accuracy 
                for a in recent_analyses[:self.thresholds.consecutive_poor_predictions]
            )
            
            if consecutive_poor:
                avg_accuracy = statistics.mean(
                    a.accuracy_percentage for a in recent_analyses[:self.thresholds.consecutive_poor_predictions]
                )
                
                triggers.append(RetrainingTrigger(
                    trigger_type='consecutive_failures',
                    trigger_value=avg_accuracy,
                    threshold_value=self.thresholds.min_accuracy,
                    description=f"{self.thresholds.consecutive_poor_predictions} predições ruins consecutivas (média: {avg_accuracy:.1f}%)",
                    timestamp=datetime.now(),
                    severity='critical'
                ))
        
        return triggers
    
    def _handle_retraining_trigger(self, trigger: RetrainingTrigger):
        """Processa um trigger de retreinamento"""
        with self.lock:
            # Adiciona ao histórico
            self.trigger_history.append(trigger)
            
            # Executa callbacks
            for callback in self.trigger_callbacks:
                try:
                    callback(trigger)
                except Exception as e:
                    logger.error(f"Erro ao executar callback de trigger: {e}")
            
            # Verifica se deve retreinar automaticamente
            if trigger.auto_retrain and self._should_auto_retrain(trigger):
                self._schedule_retraining(trigger)
            
            logger.info(f"Trigger de retreinamento: {trigger.description} (severidade: {trigger.severity})")
    
    def _should_auto_retrain(self, trigger: RetrainingTrigger) -> bool:
        """Verifica se deve executar retreinamento automático"""
        # Verifica cooldown
        if self.retraining_history:
            last_retraining = max(self.retraining_history, key=lambda x: x.start_time)
            cooldown_hours = 6  # Configurável
            
            if (datetime.now() - last_retraining.start_time).total_seconds() < cooldown_hours * 3600:
                logger.info(f"Retreinamento em cooldown, aguardando {cooldown_hours}h")
                return False
        
        # Verifica severidade
        if trigger.severity in ['high', 'critical']:
            return True
        
        # Para severidade média, verifica se há múltiplos triggers recentes
        recent_triggers = [
            t for t in self.trigger_history 
            if (datetime.now() - t.timestamp).total_seconds() < 3600  # Última hora
        ]
        
        return len(recent_triggers) >= 2
    
    def _schedule_retraining(self, trigger: RetrainingTrigger):
        """Agenda retreinamento baseado no trigger"""
        if not self.training_system:
            logger.error("Sistema de treinamento não disponível")
            return
        
        try:
            # Adiciona retreinamento à fila do sistema automatizado
            self.training_system.adicionar_treinamento_fila('performance_trigger')
            
            # Inicia retreinamento em thread separada
            retraining_thread = threading.Thread(
                target=self._execute_retraining,
                args=(trigger,),
                daemon=True
            )
            retraining_thread.start()
            
            logger.info(f"Retreinamento agendado devido a: {trigger.description}")
            
        except Exception as e:
            logger.error(f"Erro ao agendar retreinamento: {e}")
    
    def _execute_retraining(self, trigger: RetrainingTrigger):
        """Executa retreinamento"""
        result = RetrainingResult(
            trigger_id=f"{trigger.trigger_type}_{trigger.timestamp.isoformat()}",
            start_time=datetime.now(),
            end_time=None,
            success=False,
            models_retrained=[],
            performance_before={},
            performance_after={},
            improvement_metrics={}
        )
        
        try:
            # Coleta métricas antes do retreinamento
            result.performance_before = self._collect_current_performance()
            
            # Executa retreinamento
            if self.training_system:
                training_result = self.training_system.executar_treinamento_completo()
                
                result.success = training_result.get('sucesso', False)
                result.models_retrained = training_result.get('modelos_treinados', [])
                
                if not result.success:
                    result.error_message = '; '.join(training_result.get('erros', []))
            
            # Aguarda um pouco para que novos dados sejam processados
            time.sleep(60)
            
            # Coleta métricas após retreinamento
            result.performance_after = self._collect_current_performance()
            
            # Calcula melhorias
            result.improvement_metrics = self._calculate_improvements(
                result.performance_before, 
                result.performance_after
            )
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Erro durante retreinamento: {e}")
        
        finally:
            result.end_time = datetime.now()
            
            # Adiciona ao histórico
            with self.lock:
                self.retraining_history.append(result)
            
            # Executa callbacks
            for callback in self.retraining_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Erro ao executar callback de retreinamento: {e}")
            
            # Log do resultado
            if result.success:
                logger.info(f"Retreinamento concluído com sucesso. Modelos: {result.models_retrained}")
                if result.improvement_metrics:
                    logger.info(f"Melhorias: {result.improvement_metrics}")
            else:
                logger.error(f"Retreinamento falhou: {result.error_message}")
    
    def _collect_current_performance(self) -> Dict[str, float]:
        """Coleta métricas atuais de performance"""
        try:
            # Métricas do accuracy tracker
            accuracy_summary = accuracy_tracker.get_accuracy_summary(7)
            
            # Métricas do metrics service
            current_stats = metrics_service.get_current_stats()
            
            return {
                'average_accuracy': accuracy_summary['period_summary']['average_accuracy'],
                'total_predictions': accuracy_summary['period_summary']['total_predictions'],
                'confidence_correlation': accuracy_summary['period_summary']['confidence_correlation'],
                'prediction_rate': current_stats.predictions_per_hour,
                'error_rate': current_stats.error_rate
            }
            
        except Exception as e:
            logger.error(f"Erro ao coletar métricas atuais: {e}")
            return {}
    
    def _calculate_improvements(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """Calcula melhorias entre métricas antes e depois"""
        improvements = {}
        
        for metric in before.keys():
            if metric in after:
                if before[metric] != 0:
                    improvement = ((after[metric] - before[metric]) / before[metric]) * 100
                    improvements[f"{metric}_improvement_percent"] = improvement
                
                improvements[f"{metric}_absolute_change"] = after[metric] - before[metric]
        
        return improvements
    
    def start_monitoring(self):
        """Inicia monitoramento automático"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoramento de performance iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento automático"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Monitoramento de performance parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring_active:
            try:
                # Avalia necessidade de retreinamento baseado em tendências
                self._evaluate_periodic_retraining()
                
                # Aguarda próxima verificação (30 minutos por padrão)
                time.sleep(30 * 60)
                
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(5 * 60)  # Aguarda 5 minutos em caso de erro
    
    def _evaluate_periodic_retraining(self):
        """Avalia periodicamente se é necessário retreinamento"""
        try:
            # Calcula tendência atual
            trend = accuracy_tracker.calculate_accuracy_trend(self.thresholds.evaluation_period_days)
            
            if trend.total_predictions < self.thresholds.min_predictions_for_evaluation:
                return
            
            # Verifica se a tendência indica necessidade de retreinamento
            triggers = []
            
            if trend.trend_direction == 'declining':
                triggers.append(RetrainingTrigger(
                    trigger_type='declining_trend',
                    trigger_value=trend.average_accuracy,
                    threshold_value=self.thresholds.min_accuracy,
                    description=f"Tendência de declínio detectada (acurácia média: {trend.average_accuracy:.1f}%)",
                    timestamp=datetime.now(),
                    severity='medium',
                    auto_retrain=trend.average_accuracy < self.thresholds.min_accuracy
                ))
            
            # Processa triggers encontrados
            for trigger in triggers:
                self._handle_retraining_trigger(trigger)
                
        except Exception as e:
            logger.error(f"Erro na avaliação periódica: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Retorna status da integração"""
        return {
            'monitoring_active': self.monitoring_active,
            'training_system_available': self.training_system is not None,
            'thresholds': asdict(self.thresholds),
            'trigger_history_count': len(self.trigger_history),
            'retraining_history_count': len(self.retraining_history),
            'recent_triggers': [
                {
                    'type': t.trigger_type,
                    'severity': t.severity,
                    'timestamp': t.timestamp.isoformat(),
                    'description': t.description
                }
                for t in self.trigger_history[-5:]  # Últimos 5 triggers
            ],
            'recent_retrainings': [
                {
                    'trigger_id': r.trigger_id,
                    'success': r.success,
                    'start_time': r.start_time.isoformat(),
                    'models_retrained': r.models_retrained,
                    'improvement_metrics': r.improvement_metrics
                }
                for r in self.retraining_history[-3:]  # Últimos 3 retreinamentos
            ]
        }
    
    def force_retraining(self, reason: str = "Manual trigger") -> bool:
        """Força retreinamento manual"""
        trigger = RetrainingTrigger(
            trigger_type='manual',
            trigger_value=0.0,
            threshold_value=0.0,
            description=reason,
            timestamp=datetime.now(),
            severity='high',
            auto_retrain=True
        )
        
        self._handle_retraining_trigger(trigger)
        return True

# Instância global da integração
retraining_integration = RetrainingIntegration()

# Funções de conveniência
def start_integration_monitoring():
    """Inicia monitoramento integrado"""
    retraining_integration.start_monitoring()

def stop_integration_monitoring():
    """Para monitoramento integrado"""
    retraining_integration.stop_monitoring()

def get_integration_status() -> Dict[str, Any]:
    """Obtém status da integração"""
    return retraining_integration.get_integration_status()

def force_retraining(reason: str = "Manual trigger") -> bool:
    """Força retreinamento manual"""
    return retraining_integration.force_retraining(reason)

if __name__ == "__main__":
    # Teste da integração
    print("Testando integração de retreinamento...")
    
    # Inicia monitoramento
    start_integration_monitoring()
    
    # Mostra status
    status = get_integration_status()
    print(f"Status: Monitoramento ativo = {status['monitoring_active']}")
    print(f"Sistema de treinamento disponível = {status['training_system_available']}")
    
    # Simula trigger manual
    force_retraining("Teste de integração")
    
    # Para monitoramento
    time.sleep(5)
    stop_integration_monitoring()
    
    print("Teste de integração concluído!")