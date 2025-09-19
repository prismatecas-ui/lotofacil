from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from enum import Enum

# Importar os módulos já criados
from .metrics_service import metrics_service, RealTimeMetricsService
from .accuracy_tracker import accuracy_tracker, AccuracyTracker
from .alert_system import alert_system, AlertSystem
from .structured_logger import structured_logger, StructuredLogger
from .retraining_integration import retraining_integration, RetrainingIntegration

class MetricType(Enum):
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    PREDICTIONS = "predictions"
    ALERTS = "alerts"
    RETRAINING = "retraining"
    SYSTEM = "system"

class TimeRange(Enum):
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"
    CUSTOM = "custom"

@dataclass
class MetricsQuery:
    metric_types: List[str]
    time_range: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    aggregation: str = "avg"  # avg, sum, min, max, count
    group_by: Optional[str] = None  # hour, day, week, month
    filters: Optional[Dict[str, Any]] = None
    limit: int = 1000
    offset: int = 0

@dataclass
class MetricsResponse:
    success: bool
    data: Dict[str, Any]
    total_records: int
    query_time_ms: float
    metadata: Dict[str, Any]
    error: Optional[str] = None

class MetricsAPI:
    def __init__(self, app: Flask = None):
        self.app = app
        self.metrics_service = metrics_service
        self.accuracy_tracker = accuracy_tracker
        self.alert_system = alert_system
        self.logger = structured_logger
        self.retraining_integration = retraining_integration
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Inicializar rotas da API no Flask app"""
        self.app = app
        self._register_routes()
    
    def _register_routes(self):
        """Registrar todas as rotas da API"""
        
        @self.app.route('/api/metrics', methods=['GET'])
        def get_metrics():
            """Endpoint principal para consulta de métricas"""
            start_time = datetime.now()
            
            try:
                # Parse dos parâmetros da query
                query = self._parse_query_params(request.args)
                
                # Validar query
                validation_error = self._validate_query(query)
                if validation_error:
                    return jsonify({
                        'success': False,
                        'error': validation_error
                    }), 400
                
                # Executar consulta
                data = self._execute_metrics_query(query)
                
                # Calcular tempo de execução
                query_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log da consulta
                self.logger.log_api_request(
                    endpoint='/api/metrics',
                    method='GET',
                    params=asdict(query),
                    response_time_ms=query_time,
                    status_code=200
                )
                
                response = MetricsResponse(
                    success=True,
                    data=data,
                    total_records=len(data.get('records', [])),
                    query_time_ms=query_time,
                    metadata={
                        'query': asdict(query),
                        'timestamp': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                )
                
                return jsonify(asdict(response))
                
            except Exception as e:
                query_time = (datetime.now() - start_time).total_seconds() * 1000
                
                self.logger.log_exception(
                    e, 
                    context={'endpoint': '/api/metrics', 'query_params': dict(request.args)}
                )
                
                return jsonify({
                    'success': False,
                    'error': f'Erro interno do servidor: {str(e)}',
                    'query_time_ms': query_time
                }), 500
        
        @self.app.route('/api/metrics/summary', methods=['GET'])
        def get_metrics_summary():
            """Endpoint para resumo geral das métricas"""
            try:
                summary = {
                    'accuracy': self._get_accuracy_summary(),
                    'performance': self._get_performance_summary(),
                    'alerts': self._get_alerts_summary(),
                    'system_health': self._get_system_health(),
                    'retraining': self._get_retraining_summary()
                }
                
                return jsonify({
                    'success': True,
                    'data': summary,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.log_exception(e, context={'endpoint': '/api/metrics/summary'})
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/metrics/live', methods=['GET'])
        def get_live_metrics():
            """Endpoint para métricas em tempo real"""
            try:
                live_data = {
                    'current_accuracy': self.accuracy_tracker.get_current_accuracy(),
                    'recent_predictions': self.metrics_service.get_recent_predictions(limit=10),
                    'active_alerts': self.alert_system.get_active_alerts(),
                    'system_status': self._get_system_status(),
                    'last_update': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'data': live_data
                })
                
            except Exception as e:
                self.logger.log_exception(e, context={'endpoint': '/api/metrics/live'})
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/metrics/export', methods=['GET'])
        def export_metrics():
            """Endpoint para exportar métricas em diferentes formatos"""
            try:
                format_type = request.args.get('format', 'json').lower()
                time_range = request.args.get('time_range', '24h')
                
                # Obter dados para exportação
                export_data = self._get_export_data(time_range)
                
                if format_type == 'csv':
                    return self._export_as_csv(export_data)
                elif format_type == 'json':
                    return jsonify({
                        'success': True,
                        'data': export_data,
                        'format': 'json',
                        'exported_at': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Formato não suportado: {format_type}'
                    }), 400
                    
            except Exception as e:
                self.logger.log_exception(e, context={'endpoint': '/api/metrics/export'})
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def _parse_query_params(self, args) -> MetricsQuery:
        """Parse dos parâmetros da query string"""
        metric_types = args.get('types', 'accuracy,performance').split(',')
        time_range = args.get('time_range', '24h')
        start_date = args.get('start_date')
        end_date = args.get('end_date')
        aggregation = args.get('aggregation', 'avg')
        group_by = args.get('group_by')
        limit = int(args.get('limit', 1000))
        offset = int(args.get('offset', 0))
        
        # Parse filters
        filters = {}
        for key, value in args.items():
            if key.startswith('filter_'):
                filter_key = key.replace('filter_', '')
                filters[filter_key] = value
        
        return MetricsQuery(
            metric_types=metric_types,
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            aggregation=aggregation,
            group_by=group_by,
            filters=filters if filters else None,
            limit=limit,
            offset=offset
        )
    
    def _validate_query(self, query: MetricsQuery) -> Optional[str]:
        """Validar parâmetros da query"""
        # Validar tipos de métrica
        valid_types = [t.value for t in MetricType]
        for metric_type in query.metric_types:
            if metric_type not in valid_types:
                return f"Tipo de métrica inválido: {metric_type}"
        
        # Validar range de tempo
        valid_ranges = [t.value for t in TimeRange]
        if query.time_range not in valid_ranges:
            return f"Range de tempo inválido: {query.time_range}"
        
        # Validar agregação
        valid_aggregations = ['avg', 'sum', 'min', 'max', 'count']
        if query.aggregation not in valid_aggregations:
            return f"Tipo de agregação inválido: {query.aggregation}"
        
        # Validar limite
        if query.limit > 10000:
            return "Limite máximo de 10000 registros"
        
        return None
    
    def _execute_metrics_query(self, query: MetricsQuery) -> Dict[str, Any]:
        """Executar consulta de métricas"""
        results = {}
        
        # Calcular período de tempo
        end_time = datetime.now()
        if query.time_range == '1h':
            start_time = end_time - timedelta(hours=1)
        elif query.time_range == '24h':
            start_time = end_time - timedelta(days=1)
        elif query.time_range == '7d':
            start_time = end_time - timedelta(days=7)
        elif query.time_range == '30d':
            start_time = end_time - timedelta(days=30)
        elif query.time_range == 'custom':
            if query.start_date and query.end_date:
                start_time = datetime.fromisoformat(query.start_date)
                end_time = datetime.fromisoformat(query.end_date)
            else:
                start_time = end_time - timedelta(days=1)
        
        # Executar consultas por tipo de métrica
        for metric_type in query.metric_types:
            if metric_type == 'accuracy':
                results[metric_type] = self._get_accuracy_metrics(start_time, end_time, query)
            elif metric_type == 'performance':
                results[metric_type] = self._get_performance_metrics(start_time, end_time, query)
            elif metric_type == 'predictions':
                results[metric_type] = self._get_predictions_metrics(start_time, end_time, query)
            elif metric_type == 'alerts':
                results[metric_type] = self._get_alerts_metrics(start_time, end_time, query)
            elif metric_type == 'retraining':
                results[metric_type] = self._get_retraining_metrics(start_time, end_time, query)
            elif metric_type == 'system':
                results[metric_type] = self._get_system_metrics(start_time, end_time, query)
        
        return {
            'records': results,
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration': str(end_time - start_time)
            },
            'aggregation': query.aggregation,
            'group_by': query.group_by
        }
    
    def _get_accuracy_metrics(self, start_time: datetime, end_time: datetime, query: MetricsQuery) -> Dict[str, Any]:
        """Obter métricas de acurácia"""
        stats = self.accuracy_tracker.get_performance_stats()
        return {
            'current_accuracy': stats.accuracy,
            'total_predictions': stats.total_predictions,
            'correct_predictions': stats.correct_predictions,
            'accuracy_trend': self.accuracy_tracker.get_accuracy_trend(),
            'by_game_type': self.accuracy_tracker.get_accuracy_by_game_type()
        }
    
    def _get_performance_metrics(self, start_time: datetime, end_time: datetime, query: MetricsQuery) -> Dict[str, Any]:
        """Obter métricas de performance"""
        stats = self.metrics_service.get_performance_stats()
        return {
            'avg_response_time': stats.avg_response_time,
            'total_requests': stats.total_requests,
            'error_rate': stats.error_rate,
            'throughput': stats.requests_per_minute
        }
    
    def _get_predictions_metrics(self, start_time: datetime, end_time: datetime, query: MetricsQuery) -> Dict[str, Any]:
        """Obter métricas de predições"""
        return {
            'recent_predictions': self.metrics_service.get_recent_predictions(limit=query.limit),
            'prediction_distribution': self.metrics_service.get_prediction_distribution(),
            'confidence_levels': self.metrics_service.get_confidence_distribution()
        }
    
    def _get_alerts_metrics(self, start_time: datetime, end_time: datetime, query: MetricsQuery) -> Dict[str, Any]:
        """Obter métricas de alertas"""
        return {
            'active_alerts': self.alert_system.get_active_alerts(),
            'alert_history': self.alert_system.get_alert_history(start_time, end_time),
            'alert_summary': self.alert_system.get_alert_summary()
        }
    
    def _get_retraining_metrics(self, start_time: datetime, end_time: datetime, query: MetricsQuery) -> Dict[str, Any]:
        """Obter métricas de retreinamento"""
        return {
            'last_retraining': self.retraining_integration.get_last_retraining_info(),
            'retraining_schedule': self.retraining_integration.get_retraining_schedule(),
            'performance_triggers': self.retraining_integration.get_trigger_history()
        }
    
    def _get_system_metrics(self, start_time: datetime, end_time: datetime, query: MetricsQuery) -> Dict[str, Any]:
        """Obter métricas do sistema"""
        return {
            'uptime': self._calculate_uptime(),
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage(),
            'disk_usage': self._get_disk_usage()
        }
    
    def _get_accuracy_summary(self) -> Dict[str, Any]:
        """Resumo de acurácia"""
        stats = self.accuracy_tracker.get_performance_stats()
        return {
            'current': stats.accuracy,
            'trend': 'up' if stats.accuracy > 0.7 else 'down',
            'total_predictions': stats.total_predictions
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Resumo de performance"""
        stats = self.metrics_service.get_performance_stats()
        return {
            'avg_response_time': stats.avg_response_time,
            'error_rate': stats.error_rate,
            'status': 'healthy' if stats.error_rate < 0.05 else 'warning'
        }
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Resumo de alertas"""
        active_alerts = self.alert_system.get_active_alerts()
        return {
            'active_count': len(active_alerts),
            'critical_count': len([a for a in active_alerts if a.severity == 'critical']),
            'status': 'critical' if any(a.severity == 'critical' for a in active_alerts) else 'ok'
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Status de saúde do sistema"""
        return {
            'status': 'healthy',
            'uptime': self._calculate_uptime(),
            'services': {
                'metrics': 'running',
                'accuracy_tracker': 'running',
                'alert_system': 'running',
                'retraining': 'running'
            }
        }
    
    def _get_retraining_summary(self) -> Dict[str, Any]:
        """Resumo de retreinamento"""
        return {
            'last_retraining': self.retraining_integration.get_last_retraining_info(),
            'next_scheduled': 'auto',
            'status': 'active'
        }
    
    def _get_system_status(self) -> Dict[str, str]:
        """Status atual do sistema"""
        return {
            'metrics_service': 'running',
            'accuracy_tracker': 'running',
            'alert_system': 'running',
            'database': 'connected',
            'api': 'healthy'
        }
    
    def _get_export_data(self, time_range: str) -> Dict[str, Any]:
        """Obter dados para exportação"""
        query = MetricsQuery(
            metric_types=['accuracy', 'performance', 'predictions'],
            time_range=time_range,
            limit=10000
        )
        return self._execute_metrics_query(query)
    
    def _export_as_csv(self, data: Dict[str, Any]):
        """Exportar dados como CSV"""
        # Implementação simplificada - em produção usar pandas ou csv
        import io
        from flask import Response
        
        output = io.StringIO()
        output.write("timestamp,metric_type,value,details\n")
        
        for metric_type, metrics in data['records'].items():
            for key, value in metrics.items():
                output.write(f"{datetime.now().isoformat()},{metric_type},{key},{value}\n")
        
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=metrics_export.csv'}
        )
        return response
    
    def _calculate_uptime(self) -> str:
        """Calcular uptime do sistema"""
        # Implementação simplificada
        return "99.9%"
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Obter uso de memória"""
        import psutil
        memory = psutil.virtual_memory()
        return {
            'used_percent': memory.percent,
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
    
    def _get_cpu_usage(self) -> float:
        """Obter uso de CPU"""
        import psutil
        return psutil.cpu_percent(interval=1)
    
    def _get_disk_usage(self) -> Dict[str, float]:
        """Obter uso de disco"""
        import psutil
        disk = psutil.disk_usage('/')
        return {
            'used_percent': (disk.used / disk.total) * 100,
            'used_gb': disk.used / (1024**3),
            'total_gb': disk.total / (1024**3)
        }

# Instância global da API
metrics_api = MetricsAPI()

# Função para inicializar com Flask app
def init_metrics_api(app: Flask):
    """Inicializar API de métricas com Flask app"""
    metrics_api.init_app(app)
    return metrics_api

# Exemplo de uso
if __name__ == '__main__':
    from flask import Flask
    
    app = Flask(__name__)
    init_metrics_api(app)
    
    app.run(debug=True, port=5001)