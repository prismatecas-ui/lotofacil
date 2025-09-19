#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard de Performance para Monitoramento de Predições
Interface web para visualização de métricas em tempo real
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
import threading
import time
from dataclasses import asdict

# Importa sistemas de métricas
from .metrics_service import metrics_service, PredictionMetric
from .accuracy_tracker import accuracy_tracker, AccuracyAnalysis
from .retraining_integration import retraining_integration

logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Dashboard web para monitoramento de performance"""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5001):
        self.host = host
        self.port = port
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.running = False
        self.server_thread = None
        
        # Configuração do Flask
        self.app.config['SECRET_KEY'] = 'dashboard-performance-key'
        self.app.config['JSON_AS_ASCII'] = False
        
        # Registra rotas
        self._register_routes()
        
        # Cria diretórios necessários
        self._create_directories()
        
        # Cria arquivos de template e assets
        self._create_templates()
        self._create_static_files()
    
    def _create_directories(self):
        """Cria diretórios necessários para o dashboard"""
        base_path = Path(__file__).parent
        
        # Cria diretórios
        (base_path / 'templates').mkdir(exist_ok=True)
        (base_path / 'static' / 'css').mkdir(parents=True, exist_ok=True)
        (base_path / 'static' / 'js').mkdir(parents=True, exist_ok=True)
    
    def _register_routes(self):
        """Registra rotas do dashboard"""
        
        @self.app.route('/')
        def dashboard_home():
            """Página principal do dashboard"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics/current')
        def get_current_metrics():
            """API: Métricas atuais"""
            try:
                # Métricas do sistema
                current_stats = metrics_service.get_current_stats()
                
                # Resumo de acurácia
                accuracy_summary = accuracy_tracker.get_accuracy_summary(7)
                
                # Status da integração
                integration_status = retraining_integration.get_integration_status()
                
                return jsonify({
                    'success': True,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'predictions_per_hour': current_stats.predictions_per_hour,
                        'average_confidence': current_stats.average_confidence,
                        'error_rate': current_stats.error_rate,
                        'total_predictions': current_stats.total_predictions,
                        'accuracy_percentage': accuracy_summary['period_summary']['average_accuracy'],
                        'confidence_correlation': accuracy_summary['period_summary']['confidence_correlation'],
                        'monitoring_active': integration_status['monitoring_active']
                    }
                })
                
            except Exception as e:
                logger.error(f"Erro ao obter métricas atuais: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/metrics/history')
        def get_metrics_history():
            """API: Histórico de métricas"""
            try:
                days = request.args.get('days', 7, type=int)
                
                # Histórico de acurácia
                accuracy_history = accuracy_tracker.get_accuracy_history(limit=days * 5)
                
                # Métricas recentes
                recent_metrics = metrics_service.get_recent_metrics(hours=days * 24)
                
                return jsonify({
                    'success': True,
                    'period_days': days,
                    'accuracy_history': [
                        {
                            'timestamp': analysis.timestamp.isoformat(),
                            'accuracy': analysis.accuracy_percentage,
                            'confidence': analysis.average_confidence,
                            'total_predictions': analysis.total_predictions
                        }
                        for analysis in accuracy_history
                    ],
                    'metrics_history': [
                        {
                            'timestamp': metric.timestamp.isoformat(),
                            'confidence': metric.confidence_score,
                            'processing_time': metric.processing_time_ms,
                            'success': metric.success
                        }
                        for metric in recent_metrics
                    ]
                })
                
            except Exception as e:
                logger.error(f"Erro ao obter histórico: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/retraining/status')
        def get_retraining_status():
            """API: Status do sistema de retreinamento"""
            try:
                status = retraining_integration.get_integration_status()
                return jsonify({
                    'success': True,
                    'status': status
                })
                
            except Exception as e:
                logger.error(f"Erro ao obter status de retreinamento: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/retraining/trigger', methods=['POST'])
        def trigger_retraining():
            """API: Dispara retreinamento manual"""
            try:
                data = request.get_json() or {}
                reason = data.get('reason', 'Trigger manual via dashboard')
                
                success = retraining_integration.force_retraining(reason)
                
                return jsonify({
                    'success': success,
                    'message': 'Retreinamento iniciado com sucesso' if success else 'Falha ao iniciar retreinamento'
                })
                
            except Exception as e:
                logger.error(f"Erro ao disparar retreinamento: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/alerts/recent')
        def get_recent_alerts():
            """API: Alertas recentes"""
            try:
                status = retraining_integration.get_integration_status()
                
                # Converte triggers em alertas
                alerts = []
                for trigger in status.get('recent_triggers', []):
                    alerts.append({
                        'id': f"{trigger['type']}_{trigger['timestamp']}",
                        'type': trigger['type'],
                        'severity': trigger['severity'],
                        'message': trigger['description'],
                        'timestamp': trigger['timestamp'],
                        'resolved': False
                    })
                
                return jsonify({
                    'success': True,
                    'alerts': alerts
                })
                
            except Exception as e:
                logger.error(f"Erro ao obter alertas: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/static/<path:filename>')
        def serve_static(filename):
            """Serve arquivos estáticos"""
            return send_from_directory('static', filename)
    
    def _create_templates(self):
        """Cria templates HTML do dashboard"""
        template_path = Path(__file__).parent / 'templates' / 'dashboard.html'
        
        html_content = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Performance - Lotofácil AI</title>
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>🎯 Dashboard de Performance - Lotofácil AI</h1>
            <div class="status-indicator" id="connectionStatus">
                <span class="status-dot"></span>
                <span class="status-text">Conectando...</span>
            </div>
        </header>

        <main class="dashboard-main">
            <!-- Métricas Principais -->
            <section class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <h3>Acurácia Atual</h3>
                        <span class="metric-icon">🎯</span>
                    </div>
                    <div class="metric-value" id="accuracyValue">--</div>
                    <div class="metric-change" id="accuracyChange">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <h3>Predições/Hora</h3>
                        <span class="metric-icon">⚡</span>
                    </div>
                    <div class="metric-value" id="predictionsValue">--</div>
                    <div class="metric-change" id="predictionsChange">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <h3>Confiança Média</h3>
                        <span class="metric-icon">📊</span>
                    </div>
                    <div class="metric-value" id="confidenceValue">--</div>
                    <div class="metric-change" id="confidenceChange">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <h3>Taxa de Erro</h3>
                        <span class="metric-icon">⚠️</span>
                    </div>
                    <div class="metric-value" id="errorRateValue">--</div>
                    <div class="metric-change" id="errorRateChange">--</div>
                </div>
            </section>

            <!-- Gráficos -->
            <section class="charts-grid">
                <div class="chart-container">
                    <h3>Histórico de Acurácia</h3>
                    <canvas id="accuracyChart"></canvas>
                </div>

                <div class="chart-container">
                    <h3>Confiança vs Tempo</h3>
                    <canvas id="confidenceChart"></canvas>
                </div>
            </section>

            <!-- Alertas e Sistema de Retreinamento -->
            <section class="alerts-retraining">
                <div class="alerts-panel">
                    <h3>🚨 Alertas Recentes</h3>
                    <div id="alertsList" class="alerts-list">
                        <div class="no-alerts">Nenhum alerta ativo</div>
                    </div>
                </div>

                <div class="retraining-panel">
                    <h3>🔄 Sistema de Retreinamento</h3>
                    <div class="retraining-status" id="retrainingStatus">
                        <div class="status-item">
                            <span class="label">Monitoramento:</span>
                            <span class="value" id="monitoringStatus">--</span>
                        </div>
                        <div class="status-item">
                            <span class="label">Último Retreinamento:</span>
                            <span class="value" id="lastRetraining">--</span>
                        </div>
                        <div class="status-item">
                            <span class="label">Triggers Recentes:</span>
                            <span class="value" id="recentTriggers">--</span>
                        </div>
                    </div>
                    <button id="manualRetrainBtn" class="retrain-button">
                        🚀 Retreinamento Manual
                    </button>
                </div>
            </section>
        </main>
    </div>

    <script src="/static/js/dashboard.js"></script>
</body>
</html>
        '''
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _create_static_files(self):
        """Cria arquivos CSS e JavaScript"""
        # CSS
        css_path = Path(__file__).parent / 'static' / 'css' / 'dashboard.css'
        css_content = '''
/* Dashboard de Performance - Estilos */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.dashboard-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.dashboard-header {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

.dashboard-header h1 {
    color: #2c3e50;
    font-size: 2rem;
    font-weight: 700;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #95a5a6;
    animation: pulse 2s infinite;
}

.status-dot.connected {
    background: #27ae60;
}

.status-dot.error {
    background: #e74c3c;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.metric-header h3 {
    color: #2c3e50;
    font-size: 1.1rem;
    font-weight: 600;
}

.metric-icon {
    font-size: 1.5rem;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #3498db;
    margin-bottom: 8px;
}

.metric-change {
    font-size: 0.9rem;
    font-weight: 500;
}

.metric-change.positive {
    color: #27ae60;
}

.metric-change.negative {
    color: #e74c3c;
}

.charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 30px;
    margin-bottom: 30px;
}

.chart-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

.chart-container h3 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.2rem;
    font-weight: 600;
}

.alerts-retraining {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
}

.alerts-panel, .retraining-panel {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

.alerts-panel h3, .retraining-panel h3 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.2rem;
    font-weight: 600;
}

.alerts-list {
    max-height: 200px;
    overflow-y: auto;
}

.alert-item {
    padding: 12px;
    margin-bottom: 10px;
    border-radius: 8px;
    border-left: 4px solid;
}

.alert-item.high {
    background: #fdf2f2;
    border-color: #e74c3c;
}

.alert-item.medium {
    background: #fef9e7;
    border-color: #f39c12;
}

.alert-item.low {
    background: #f0f9ff;
    border-color: #3498db;
}

.no-alerts {
    text-align: center;
    color: #7f8c8d;
    padding: 20px;
    font-style: italic;
}

.retraining-status {
    margin-bottom: 20px;
}

.status-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #ecf0f1;
}

.status-item:last-child {
    border-bottom: none;
}

.status-item .label {
    font-weight: 600;
    color: #2c3e50;
}

.status-item .value {
    color: #7f8c8d;
}

.retrain-button {
    width: 100%;
    padding: 12px 20px;
    background: linear-gradient(135deg, #3498db, #2980b9);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.retrain-button:hover {
    background: linear-gradient(135deg, #2980b9, #1f4e79);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
}

.retrain-button:disabled {
    background: #bdc3c7;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

@media (max-width: 768px) {
    .dashboard-header {
        flex-direction: column;
        gap: 15px;
        text-align: center;
    }
    
    .dashboard-header h1 {
        font-size: 1.5rem;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    
    .charts-grid {
        grid-template-columns: 1fr;
    }
    
    .alerts-retraining {
        grid-template-columns: 1fr;
    }
}
        '''
        
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        # JavaScript
        js_path = Path(__file__).parent / 'static' / 'js' / 'dashboard.js'
        js_content = '''
// Dashboard de Performance - JavaScript
class PerformanceDashboard {
    constructor() {
        this.charts = {};
        this.updateInterval = null;
        this.connectionStatus = document.getElementById('connectionStatus');
        
        this.init();
    }
    
    init() {
        this.setupCharts();
        this.setupEventListeners();
        this.startAutoUpdate();
        this.updateMetrics();
    }
    
    setupCharts() {
        // Gráfico de Acurácia
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
        this.charts.accuracy = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Acurácia (%)',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Gráfico de Confiança
        const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
        this.charts.confidence = new Chart(confidenceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Confiança Média',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    setupEventListeners() {
        // Botão de retreinamento manual
        document.getElementById('manualRetrainBtn').addEventListener('click', () => {
            this.triggerManualRetraining();
        });
    }
    
    startAutoUpdate() {
        this.updateInterval = setInterval(() => {
            this.updateMetrics();
            this.updateCharts();
            this.updateAlerts();
            this.updateRetrainingStatus();
        }, 30000); // Atualiza a cada 30 segundos
    }
    
    async updateMetrics() {
        try {
            const response = await fetch('/api/metrics/current');
            const data = await response.json();
            
            if (data.success) {
                this.updateConnectionStatus(true);
                this.displayMetrics(data.metrics);
            } else {
                this.updateConnectionStatus(false, data.error);
            }
        } catch (error) {
            console.error('Erro ao atualizar métricas:', error);
            this.updateConnectionStatus(false, error.message);
        }
    }
    
    async updateCharts() {
        try {
            const response = await fetch('/api/metrics/history?days=7');
            const data = await response.json();
            
            if (data.success) {
                this.updateAccuracyChart(data.accuracy_history);
                this.updateConfidenceChart(data.metrics_history);
            }
        } catch (error) {
            console.error('Erro ao atualizar gráficos:', error);
        }
    }
    
    async updateAlerts() {
        try {
            const response = await fetch('/api/alerts/recent');
            const data = await response.json();
            
            if (data.success) {
                this.displayAlerts(data.alerts);
            }
        } catch (error) {
            console.error('Erro ao atualizar alertas:', error);
        }
    }
    
    async updateRetrainingStatus() {
        try {
            const response = await fetch('/api/retraining/status');
            const data = await response.json();
            
            if (data.success) {
                this.displayRetrainingStatus(data.status);
            }
        } catch (error) {
            console.error('Erro ao atualizar status de retreinamento:', error);
        }
    }
    
    updateConnectionStatus(connected, error = null) {
        const statusDot = this.connectionStatus.querySelector('.status-dot');
        const statusText = this.connectionStatus.querySelector('.status-text');
        
        if (connected) {
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Conectado';
        } else {
            statusDot.className = 'status-dot error';
            statusText.textContent = error ? `Erro: ${error}` : 'Desconectado';
        }
    }
    
    displayMetrics(metrics) {
        // Atualiza valores das métricas
        document.getElementById('accuracyValue').textContent = 
            metrics.accuracy_percentage ? `${metrics.accuracy_percentage.toFixed(1)}%` : '--';
        
        document.getElementById('predictionsValue').textContent = 
            metrics.predictions_per_hour ? metrics.predictions_per_hour.toFixed(0) : '--';
        
        document.getElementById('confidenceValue').textContent = 
            metrics.average_confidence ? `${(metrics.average_confidence * 100).toFixed(1)}%` : '--';
        
        document.getElementById('errorRateValue').textContent = 
            metrics.error_rate ? `${(metrics.error_rate * 100).toFixed(2)}%` : '--';
    }
    
    updateAccuracyChart(accuracyHistory) {
        const chart = this.charts.accuracy;
        
        // Prepara dados (últimos 20 pontos)
        const recentData = accuracyHistory.slice(-20);
        
        chart.data.labels = recentData.map(item => {
            const date = new Date(item.timestamp);
            return date.toLocaleDateString('pt-BR') + ' ' + date.toLocaleTimeString('pt-BR', {hour: '2-digit', minute: '2-digit'});
        });
        
        chart.data.datasets[0].data = recentData.map(item => item.accuracy);
        
        chart.update('none');
    }
    
    updateConfidenceChart(metricsHistory) {
        const chart = this.charts.confidence;
        
        // Agrupa por hora e calcula média
        const hourlyData = {};
        
        metricsHistory.forEach(item => {
            const hour = new Date(item.timestamp).toISOString().slice(0, 13);
            if (!hourlyData[hour]) {
                hourlyData[hour] = [];
            }
            hourlyData[hour].push(item.confidence);
        });
        
        const aggregatedData = Object.entries(hourlyData)
            .map(([hour, confidences]) => ({
                timestamp: hour,
                avgConfidence: confidences.reduce((a, b) => a + b, 0) / confidences.length
            }))
            .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
            .slice(-24); // Últimas 24 horas
        
        chart.data.labels = aggregatedData.map(item => {
            const date = new Date(item.timestamp + ':00:00');
            return date.toLocaleTimeString('pt-BR', {hour: '2-digit', minute: '2-digit'});
        });
        
        chart.data.datasets[0].data = aggregatedData.map(item => item.avgConfidence);
        
        chart.update('none');
    }
    
    displayAlerts(alerts) {
        const alertsList = document.getElementById('alertsList');
        
        if (alerts.length === 0) {
            alertsList.innerHTML = '<div class="no-alerts">Nenhum alerta ativo</div>';
            return;
        }
        
        alertsList.innerHTML = alerts.map(alert => `
            <div class="alert-item ${alert.severity}">
                <div class="alert-message">${alert.message}</div>
                <div class="alert-time">${new Date(alert.timestamp).toLocaleString('pt-BR')}</div>
            </div>
        `).join('');
    }
    
    displayRetrainingStatus(status) {
        document.getElementById('monitoringStatus').textContent = 
            status.monitoring_active ? '🟢 Ativo' : '🔴 Inativo';
        
        const recentRetrainings = status.recent_retrainings || [];
        if (recentRetrainings.length > 0) {
            const last = recentRetrainings[recentRetrainings.length - 1];
            document.getElementById('lastRetraining').textContent = 
                new Date(last.start_time).toLocaleString('pt-BR');
        } else {
            document.getElementById('lastRetraining').textContent = 'Nenhum';
        }
        
        document.getElementById('recentTriggers').textContent = 
            status.trigger_history_count || 0;
    }
    
    async triggerManualRetraining() {
        const button = document.getElementById('manualRetrainBtn');
        
        if (button.disabled) return;
        
        button.disabled = true;
        button.textContent = '🔄 Iniciando...';
        
        try {
            const response = await fetch('/api/retraining/trigger', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    reason: 'Retreinamento manual via dashboard'
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                button.textContent = '✅ Iniciado!';
                setTimeout(() => {
                    button.textContent = '🚀 Retreinamento Manual';
                    button.disabled = false;
                }, 3000);
            } else {
                button.textContent = '❌ Erro';
                setTimeout(() => {
                    button.textContent = '🚀 Retreinamento Manual';
                    button.disabled = false;
                }, 3000);
            }
        } catch (error) {
            console.error('Erro ao disparar retreinamento:', error);
            button.textContent = '❌ Erro';
            setTimeout(() => {
                button.textContent = '🚀 Retreinamento Manual';
                button.disabled = false;
            }, 3000);
        }
    }
    
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        Object.values(this.charts).forEach(chart => {
            chart.destroy();
        });
    }
}

// Inicializa dashboard quando a página carrega
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new PerformanceDashboard();
});

// Limpa recursos quando a página é fechada
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});
        '''
        
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write(js_content)
    
    def start(self):
        """Inicia o servidor do dashboard"""
        if self.running:
            logger.warning("Dashboard já está rodando")
            return
        
        self.running = True
        
        def run_server():
            try:
                self.app.run(
                    host=self.host,
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            except Exception as e:
                logger.error(f"Erro ao iniciar servidor do dashboard: {e}")
                self.running = False
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        logger.info(f"Dashboard iniciado em http://{self.host}:{self.port}")
    
    def stop(self):
        """Para o servidor do dashboard"""
        self.running = False
        logger.info("Dashboard parado")
    
    def get_url(self) -> str:
        """Retorna URL do dashboard"""
        return f"http://{self.host}:{self.port}"

# Instância global do dashboard
dashboard = PerformanceDashboard()

# Funções de conveniência
def start_dashboard(host: str = '127.0.0.1', port: int = 5001):
    """Inicia dashboard de performance"""
    global dashboard
    dashboard.host = host
    dashboard.port = port
    dashboard.start()
    return dashboard.get_url()

def stop_dashboard():
    """Para dashboard de performance"""
    dashboard.stop()

def get_dashboard_url() -> str:
    """Obtém URL do dashboard"""
    return dashboard.get_url()

if __name__ == "__main__":
    # Teste do dashboard
    print("Iniciando dashboard de performance...")
    
    url = start_dashboard()
    print(f"Dashboard disponível em: {url}")
    
    try:
        # Mantém o dashboard rodando
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nParando dashboard...")
        stop_dashboard()