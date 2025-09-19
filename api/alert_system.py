#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Alertas para Monitoramento de Performance
Detecta quedas de performance e envia notificações automáticas
"""

import json
import logging
import smtplib
import threading
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import requests

# Importa sistemas de métricas
from .metrics_service import metrics_service
from .accuracy_tracker import accuracy_tracker

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Níveis de severidade dos alertas"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Tipos de alertas"""
    ACCURACY_DROP = "accuracy_drop"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_CONFIDENCE = "low_confidence"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NO_PREDICTIONS = "no_predictions"

@dataclass
class AlertRule:
    """Regra de alerta"""
    id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    threshold: float
    duration_minutes: int
    enabled: bool = True
    description: str = ""
    
@dataclass
class Alert:
    """Alerta gerado"""
    id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class NotificationChannel:
    """Canal de notificação"""
    id: str
    name: str
    type: str  # email, webhook, slack, etc.
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = None
    
    def __post_init__(self):
        if self.severity_filter is None:
            self.severity_filter = list(AlertSeverity)

class AlertSystem:
    """Sistema principal de alertas"""
    
    def __init__(self, config_file: str = "config/alerts.json"):
        self.config_file = Path(config_file)
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, NotificationChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Carrega configuração
        self._load_config()
        
        # Configura regras padrão se não existirem
        if not self.rules:
            self._setup_default_rules()
        
        # Configura canais padrão se não existirem
        if not self.channels:
            self._setup_default_channels()
    
    def _load_config(self):
        """Carrega configuração de alertas"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Carrega regras
                for rule_data in config.get('rules', []):
                    rule = AlertRule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        alert_type=AlertType(rule_data['alert_type']),
                        severity=AlertSeverity(rule_data['severity']),
                        threshold=rule_data['threshold'],
                        duration_minutes=rule_data['duration_minutes'],
                        enabled=rule_data.get('enabled', True),
                        description=rule_data.get('description', '')
                    )
                    self.rules[rule.id] = rule
                
                # Carrega canais
                for channel_data in config.get('channels', []):
                    channel = NotificationChannel(
                        id=channel_data['id'],
                        name=channel_data['name'],
                        type=channel_data['type'],
                        config=channel_data['config'],
                        enabled=channel_data.get('enabled', True),
                        severity_filter=[AlertSeverity(s) for s in channel_data.get('severity_filter', [])]
                    )
                    self.channels[channel.id] = channel
                    
                logger.info(f"Configuração de alertas carregada: {len(self.rules)} regras, {len(self.channels)} canais")
                
        except Exception as e:
            logger.error(f"Erro ao carregar configuração de alertas: {e}")
    
    def _save_config(self):
        """Salva configuração de alertas"""
        try:
            # Cria diretório se não existir
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                'rules': [
                    {
                        'id': rule.id,
                        'name': rule.name,
                        'alert_type': rule.alert_type.value,
                        'severity': rule.severity.value,
                        'threshold': rule.threshold,
                        'duration_minutes': rule.duration_minutes,
                        'enabled': rule.enabled,
                        'description': rule.description
                    }
                    for rule in self.rules.values()
                ],
                'channels': [
                    {
                        'id': channel.id,
                        'name': channel.name,
                        'type': channel.type,
                        'config': channel.config,
                        'enabled': channel.enabled,
                        'severity_filter': [s.value for s in channel.severity_filter]
                    }
                    for channel in self.channels.values()
                ]
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            logger.info("Configuração de alertas salva")
            
        except Exception as e:
            logger.error(f"Erro ao salvar configuração de alertas: {e}")
    
    def _setup_default_rules(self):
        """Configura regras padrão de alertas"""
        default_rules = [
            AlertRule(
                id="accuracy_drop_critical",
                name="Queda Crítica de Acurácia",
                alert_type=AlertType.ACCURACY_DROP,
                severity=AlertSeverity.CRITICAL,
                threshold=0.70,  # Abaixo de 70%
                duration_minutes=15,
                description="Acurácia caiu abaixo de 70% por mais de 15 minutos"
            ),
            AlertRule(
                id="accuracy_drop_high",
                name="Queda Alta de Acurácia",
                alert_type=AlertType.ACCURACY_DROP,
                severity=AlertSeverity.HIGH,
                threshold=0.75,  # Abaixo de 75%
                duration_minutes=30,
                description="Acurácia caiu abaixo de 75% por mais de 30 minutos"
            ),
            AlertRule(
                id="high_error_rate",
                name="Taxa de Erro Elevada",
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=AlertSeverity.HIGH,
                threshold=0.05,  # Acima de 5%
                duration_minutes=10,
                description="Taxa de erro acima de 5% por mais de 10 minutos"
            ),
            AlertRule(
                id="low_confidence",
                name="Confiança Baixa",
                alert_type=AlertType.LOW_CONFIDENCE,
                severity=AlertSeverity.MEDIUM,
                threshold=0.60,  # Abaixo de 60%
                duration_minutes=20,
                description="Confiança média abaixo de 60% por mais de 20 minutos"
            ),
            AlertRule(
                id="no_predictions",
                name="Sem Predições",
                alert_type=AlertType.NO_PREDICTIONS,
                severity=AlertSeverity.CRITICAL,
                threshold=0,  # Nenhuma predição
                duration_minutes=5,
                description="Nenhuma predição registrada por mais de 5 minutos"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
        
        logger.info(f"Configuradas {len(default_rules)} regras padrão de alertas")
    
    def _setup_default_channels(self):
        """Configura canais padrão de notificação"""
        # Canal de log (sempre ativo)
        log_channel = NotificationChannel(
            id="log_channel",
            name="Log do Sistema",
            type="log",
            config={},
            enabled=True,
            severity_filter=list(AlertSeverity)
        )
        self.channels[log_channel.id] = log_channel
        
        # Canal de email (configurar conforme necessário)
        email_channel = NotificationChannel(
            id="email_channel",
            name="Notificações por Email",
            type="email",
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",  # Configurar
                "password": "",  # Configurar
                "from_email": "",  # Configurar
                "to_emails": []  # Configurar
            },
            enabled=False,  # Desabilitado até configurar
            severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        )
        self.channels[email_channel.id] = email_channel
        
        logger.info(f"Configurados {len(self.channels)} canais padrão")
    
    def start_monitoring(self):
        """Inicia monitoramento de alertas"""
        if self.monitoring:
            logger.warning("Monitoramento de alertas já está ativo")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Monitoramento de alertas iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento de alertas"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Monitoramento de alertas parado")
    
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring:
            try:
                self._check_all_rules()
                time.sleep(30)  # Verifica a cada 30 segundos
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(60)  # Espera mais tempo em caso de erro
    
    def _check_all_rules(self):
        """Verifica todas as regras de alerta"""
        for rule in self.rules.values():
            if rule.enabled:
                try:
                    self._check_rule(rule)
                except Exception as e:
                    logger.error(f"Erro ao verificar regra {rule.id}: {e}")
    
    def _check_rule(self, rule: AlertRule):
        """Verifica uma regra específica"""
        current_value = self._get_metric_value(rule.alert_type)
        
        if current_value is None:
            return
        
        # Verifica se a condição do alerta foi atendida
        condition_met = self._evaluate_condition(rule, current_value)
        
        alert_id = f"{rule.id}_{rule.alert_type.value}"
        
        if condition_met:
            # Verifica se já existe um alerta ativo
            if alert_id not in self.active_alerts:
                # Cria novo alerta
                alert = Alert(
                    id=alert_id,
                    rule_id=rule.id,
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    title=rule.name,
                    message=self._generate_alert_message(rule, current_value),
                    timestamp=datetime.now(),
                    metadata={
                        'current_value': current_value,
                        'threshold': rule.threshold,
                        'duration_minutes': rule.duration_minutes
                    }
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Envia notificações
                self._send_notifications(alert)
                
                logger.warning(f"Alerta ativado: {alert.title} - {alert.message}")
        else:
            # Resolve alerta se estava ativo
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                del self.active_alerts[alert_id]
                
                logger.info(f"Alerta resolvido: {alert.title}")
    
    def _get_metric_value(self, alert_type: AlertType) -> Optional[float]:
        """Obtém valor atual da métrica para o tipo de alerta"""
        try:
            if alert_type == AlertType.ACCURACY_DROP:
                summary = accuracy_tracker.get_accuracy_summary(hours=1)
                return summary.get('period_summary', {}).get('average_accuracy', 0) / 100
            
            elif alert_type == AlertType.HIGH_ERROR_RATE:
                stats = metrics_service.get_current_stats()
                return stats.error_rate
            
            elif alert_type == AlertType.LOW_CONFIDENCE:
                stats = metrics_service.get_current_stats()
                return stats.average_confidence
            
            elif alert_type == AlertType.NO_PREDICTIONS:
                stats = metrics_service.get_current_stats()
                return stats.predictions_per_hour
            
            elif alert_type == AlertType.PERFORMANCE_DEGRADATION:
                # Métrica composta baseada em múltiplos fatores
                stats = metrics_service.get_current_stats()
                accuracy_summary = accuracy_tracker.get_accuracy_summary(hours=1)
                
                accuracy_score = accuracy_summary.get('period_summary', {}).get('average_accuracy', 0) / 100
                confidence_score = stats.average_confidence
                error_penalty = 1 - stats.error_rate
                
                # Score composto (0-1)
                composite_score = (accuracy_score * 0.5 + confidence_score * 0.3 + error_penalty * 0.2)
                return composite_score
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter métrica para {alert_type}: {e}")
            return None
    
    def _evaluate_condition(self, rule: AlertRule, current_value: float) -> bool:
        """Avalia se a condição do alerta foi atendida"""
        if rule.alert_type in [AlertType.ACCURACY_DROP, AlertType.LOW_CONFIDENCE, AlertType.PERFORMANCE_DEGRADATION]:
            # Para estes tipos, alerta quando valor está ABAIXO do threshold
            return current_value < rule.threshold
        
        elif rule.alert_type in [AlertType.HIGH_ERROR_RATE]:
            # Para estes tipos, alerta quando valor está ACIMA do threshold
            return current_value > rule.threshold
        
        elif rule.alert_type == AlertType.NO_PREDICTIONS:
            # Alerta quando não há predições
            return current_value == 0
        
        return False
    
    def _generate_alert_message(self, rule: AlertRule, current_value: float) -> str:
        """Gera mensagem do alerta"""
        if rule.alert_type == AlertType.ACCURACY_DROP:
            return f"Acurácia atual: {current_value*100:.1f}% (limite: {rule.threshold*100:.1f}%)"
        
        elif rule.alert_type == AlertType.HIGH_ERROR_RATE:
            return f"Taxa de erro atual: {current_value*100:.2f}% (limite: {rule.threshold*100:.2f}%)"
        
        elif rule.alert_type == AlertType.LOW_CONFIDENCE:
            return f"Confiança atual: {current_value*100:.1f}% (limite: {rule.threshold*100:.1f}%)"
        
        elif rule.alert_type == AlertType.NO_PREDICTIONS:
            return f"Nenhuma predição registrada nas últimas {rule.duration_minutes} minutos"
        
        elif rule.alert_type == AlertType.PERFORMANCE_DEGRADATION:
            return f"Score de performance: {current_value*100:.1f}% (limite: {rule.threshold*100:.1f}%)"
        
        return f"Condição de alerta atendida: {current_value} vs {rule.threshold}"
    
    def _send_notifications(self, alert: Alert):
        """Envia notificações para os canais configurados"""
        for channel in self.channels.values():
            if not channel.enabled:
                continue
            
            if alert.severity not in channel.severity_filter:
                continue
            
            try:
                self._send_to_channel(alert, channel)
            except Exception as e:
                logger.error(f"Erro ao enviar notificação para canal {channel.id}: {e}")
    
    def _send_to_channel(self, alert: Alert, channel: NotificationChannel):
        """Envia notificação para um canal específico"""
        if channel.type == "log":
            self._send_log_notification(alert)
        
        elif channel.type == "email":
            self._send_email_notification(alert, channel)
        
        elif channel.type == "webhook":
            self._send_webhook_notification(alert, channel)
        
        else:
            logger.warning(f"Tipo de canal não suportado: {channel.type}")
    
    def _send_log_notification(self, alert: Alert):
        """Envia notificação para o log"""
        severity_emoji = {
            AlertSeverity.LOW: "ℹ️",
            AlertSeverity.MEDIUM: "⚠️",
            AlertSeverity.HIGH: "🚨",
            AlertSeverity.CRITICAL: "🔥"
        }
        
        emoji = severity_emoji.get(alert.severity, "⚠️")
        
        log_message = f"{emoji} ALERTA [{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Envia notificação por email"""
        config = channel.config
        
        if not all([config.get('username'), config.get('password'), config.get('to_emails')]):
            logger.warning("Configuração de email incompleta")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = config.get('from_email', config['username'])
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[ALERTA {alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alerta de Performance - Lotofácil AI
            
            Tipo: {alert.alert_type.value}
            Severidade: {alert.severity.value.upper()}
            Timestamp: {alert.timestamp.strftime('%d/%m/%Y %H:%M:%S')}
            
            Descrição: {alert.message}
            
            Metadados:
            {json.dumps(alert.metadata, indent=2, ensure_ascii=False)}
            
            ---
            Sistema de Monitoramento Lotofácil AI
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            
            text = msg.as_string()
            server.sendmail(config['username'], config['to_emails'], text)
            server.quit()
            
            logger.info(f"Email de alerta enviado para {len(config['to_emails'])} destinatários")
            
        except Exception as e:
            logger.error(f"Erro ao enviar email: {e}")
    
    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Envia notificação via webhook"""
        config = channel.config
        url = config.get('url')
        
        if not url:
            logger.warning("URL do webhook não configurada")
            return
        
        payload = {
            'alert_id': alert.id,
            'rule_id': alert.rule_id,
            'type': alert.alert_type.value,
            'severity': alert.severity.value,
            'title': alert.title,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'metadata': alert.metadata
        }
        
        headers = config.get('headers', {'Content-Type': 'application/json'})
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook enviado com sucesso: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Erro ao enviar webhook: {e}")
    
    def add_rule(self, rule: AlertRule):
        """Adiciona nova regra de alerta"""
        self.rules[rule.id] = rule
        self._save_config()
        logger.info(f"Regra de alerta adicionada: {rule.id}")
    
    def remove_rule(self, rule_id: str):
        """Remove regra de alerta"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self._save_config()
            logger.info(f"Regra de alerta removida: {rule_id}")
    
    def add_channel(self, channel: NotificationChannel):
        """Adiciona canal de notificação"""
        self.channels[channel.id] = channel
        self._save_config()
        logger.info(f"Canal de notificação adicionado: {channel.id}")
    
    def remove_channel(self, channel_id: str):
        """Remove canal de notificação"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            self._save_config()
            logger.info(f"Canal de notificação removido: {channel_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Retorna alertas ativos"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Retorna histórico de alertas"""
        return sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status do sistema de alertas"""
        return {
            'monitoring_active': self.monitoring,
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules.values() if r.enabled]),
            'total_channels': len(self.channels),
            'enabled_channels': len([c for c in self.channels.values() if c.enabled]),
            'active_alerts': len(self.active_alerts),
            'total_alerts_today': len([
                a for a in self.alert_history 
                if a.timestamp.date() == datetime.now().date()
            ])
        }
    
    def force_check(self) -> Dict[str, Any]:
        """Força verificação de todas as regras"""
        results = {}
        
        for rule in self.rules.values():
            if rule.enabled:
                try:
                    current_value = self._get_metric_value(rule.alert_type)
                    condition_met = self._evaluate_condition(rule, current_value) if current_value is not None else False
                    
                    results[rule.id] = {
                        'rule_name': rule.name,
                        'current_value': current_value,
                        'threshold': rule.threshold,
                        'condition_met': condition_met,
                        'status': 'ALERT' if condition_met else 'OK'
                    }
                except Exception as e:
                    results[rule.id] = {
                        'rule_name': rule.name,
                        'error': str(e),
                        'status': 'ERROR'
                    }
        
        return results

# Instância global do sistema de alertas
alert_system = AlertSystem()

# Funções de conveniência
def start_alert_monitoring():
    """Inicia monitoramento de alertas"""
    alert_system.start_monitoring()

def stop_alert_monitoring():
    """Para monitoramento de alertas"""
    alert_system.stop_monitoring()

def get_active_alerts() -> List[Alert]:
    """Obtém alertas ativos"""
    return alert_system.get_active_alerts()

def get_alert_history(limit: int = 100) -> List[Alert]:
    """Obtém histórico de alertas"""
    return alert_system.get_alert_history(limit)

def force_alert_check() -> Dict[str, Any]:
    """Força verificação de alertas"""
    return alert_system.force_check()

if __name__ == "__main__":
    # Teste do sistema de alertas
    print("Iniciando sistema de alertas...")
    
    start_alert_monitoring()
    
    try:
        # Mantém o sistema rodando
        while True:
            time.sleep(10)
            
            # Mostra status a cada 60 segundos
            status = alert_system.get_system_status()
            print(f"Status: {status['active_alerts']} alertas ativos, {status['enabled_rules']} regras ativas")
            
    except KeyboardInterrupt:
        print("\nParando sistema de alertas...")
        stop_alert_monitoring()