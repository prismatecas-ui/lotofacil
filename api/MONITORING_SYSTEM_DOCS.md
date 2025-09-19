# Sistema de Monitoramento e Métricas - Lotofácil

## Visão Geral

O Sistema de Monitoramento e Métricas foi desenvolvido para acompanhar a performance das predições do sistema Lotofácil em tempo real, fornecendo insights detalhados sobre acurácia, performance e saúde geral do sistema.

## Arquitetura do Sistema

### Componentes Principais

1. **Metrics Service** (`api/metrics_service.py`)
   - Coleta e armazena métricas em tempo real
   - Calcula estatísticas de performance
   - Gerencia histórico de predições

2. **Accuracy Tracker** (`api/accuracy_tracker.py`)
   - Monitora acurácia das predições vs resultados reais
   - Calcula tendências de acurácia
   - Analisa performance por tipo de jogo

3. **Alert System** (`api/alert_system.py`)
   - Sistema de alertas configurável
   - Notificações automáticas por email/webhook
   - Monitoramento de quedas de performance

4. **Structured Logger** (`api/structured_logger.py`)
   - Logs estruturados para análise detalhada
   - Rastreamento de performance
   - Análise de logs automatizada

5. **Retraining Integration** (`api/retraining_integration.py`)
   - Integração com sistema de retreinamento automático
   - Triggers baseados em performance
   - Agendamento inteligente de retreinamento

6. **Performance Dashboard** (`api/performance_dashboard.py`)
   - Interface web para visualização
   - Gráficos e métricas em tempo real
   - Relatórios de performance

7. **Metrics API** (`api/metrics_api.py`)
   - API REST para consulta de métricas
   - Filtros e agregações avançadas
   - Exportação de dados

## Instalação e Configuração

### Dependências

```bash
pip install flask sqlalchemy psutil pandas matplotlib seaborn
```

### Configuração Inicial

1. **Configurar Banco de Dados**
```python
# Em config.py ou .env
DATABASE_URL = "sqlite:///metrics.db"
# ou
DATABASE_URL = "mysql://user:password@localhost/lotofacil_metrics"
```

2. **Configurar Alertas**
```python
# Configuração de email (opcional)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "seu_email@gmail.com"
SMTP_PASSWORD = "sua_senha_app"

# Configuração de webhook (opcional)
WEBHOOK_URL = "https://hooks.slack.com/services/..."
```

3. **Inicializar Sistema**
```python
from api.metrics_service import metrics_service
from api.accuracy_tracker import accuracy_tracker
from api.alert_system import alert_system
from api.retraining_integration import retraining_integration

# Inicializar todos os serviços
metrics_service.start()
accuracy_tracker.start()
alert_system.start()
retraining_integration.start()
```

## Guia de Uso

### 1. Registrando Predições

```python
from api.metrics_service import metrics_service
from datetime import datetime

# Registrar uma predição
metrics_service.record_prediction(
    game_type="lotofacil",
    numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    confidence=0.85,
    model_version="v2.1",
    metadata={"algorithm": "lstm", "features": 50}
)
```

### 2. Registrando Resultados

```python
from api.accuracy_tracker import accuracy_tracker

# Registrar resultado de um sorteio
accuracy_tracker.record_result(
    game_type="lotofacil",
    draw_number=2500,
    winning_numbers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44],
    draw_date=datetime.now()
)
```

### 3. Consultando Métricas via API

#### Métricas Gerais
```bash
# Obter métricas das últimas 24 horas
curl "http://localhost:5000/api/metrics?types=accuracy,performance&time_range=24h"

# Obter resumo geral
curl "http://localhost:5000/api/metrics/summary"

# Métricas em tempo real
curl "http://localhost:5000/api/metrics/live"
```

#### Filtros Avançados
```bash
# Filtrar por período customizado
curl "http://localhost:5000/api/metrics?types=accuracy&time_range=custom&start_date=2024-01-01&end_date=2024-01-31"

# Agrupar por dia
curl "http://localhost:5000/api/metrics?types=performance&time_range=7d&group_by=day"

# Aplicar filtros específicos
curl "http://localhost:5000/api/metrics?types=predictions&filter_model_version=v2.1&filter_confidence_min=0.8"
```

#### Exportação de Dados
```bash
# Exportar como JSON
curl "http://localhost:5000/api/metrics/export?format=json&time_range=7d"

# Exportar como CSV
curl "http://localhost:5000/api/metrics/export?format=csv&time_range=30d" > metrics.csv
```

### 4. Configurando Alertas

```python
from api.alert_system import alert_system, AlertRule

# Criar regra de alerta personalizada
rule = AlertRule(
    name="Baixa Acurácia",
    metric_type="accuracy",
    condition="<",
    threshold=0.6,
    severity="warning",
    description="Acurácia abaixo de 60%"
)

alert_system.add_rule(rule)

# Configurar canal de notificação
from api.alert_system import NotificationChannel

channel = NotificationChannel(
    name="Email Admin",
    type="email",
    config={
        "to": "admin@empresa.com",
        "subject": "Alerta Sistema Lotofácil"
    }
)

alert_system.add_notification_channel(channel)
```

### 5. Acessando Dashboard

```python
from flask import Flask
from api.performance_dashboard import dashboard

app = Flask(__name__)
dashboard.init_app(app)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Acesse: `http://localhost:5000/dashboard`

### 6. Configurando Retreinamento Automático

```python
from api.retraining_integration import retraining_integration, PerformanceThreshold

# Configurar thresholds para retreinamento
threshold = PerformanceThreshold(
    accuracy_min=0.65,
    error_rate_max=0.1,
    confidence_min=0.7,
    sample_size_min=100
)

retraining_integration.configure_thresholds(threshold)

# Configurar agendamento
retraining_integration.schedule_retraining(
    frequency="weekly",
    day_of_week=0,  # Segunda-feira
    hour=2  # 02:00
)
```

## Monitoramento e Análise

### Métricas Principais

1. **Acurácia**
   - Taxa de acerto geral
   - Acurácia por tipo de jogo
   - Tendência temporal
   - Distribuição de confiança

2. **Performance**
   - Tempo de resposta médio
   - Taxa de erro
   - Throughput (requisições/minuto)
   - Uso de recursos (CPU, memória)

3. **Predições**
   - Volume de predições
   - Distribuição de números
   - Padrões identificados
   - Histórico de modelos

4. **Sistema**
   - Uptime
   - Saúde dos serviços
   - Logs de erro
   - Alertas ativos

### Análise de Logs

```python
from api.structured_logger import structured_logger

# Analisar logs do último dia
analysis = structured_logger.analyzer.analyze_period(
    start_time=datetime.now() - timedelta(days=1),
    end_time=datetime.now()
)

print(f"Erros encontrados: {analysis['error_count']}")
print(f"Performance média: {analysis['avg_performance']}")
print(f"Padrões identificados: {analysis['patterns']}")
```

### Relatórios Automatizados

```python
# Gerar relatório diário
from api.performance_dashboard import dashboard

report = dashboard.generate_daily_report()
print(report)

# Enviar relatório por email
dashboard.send_report_email(
    report=report,
    recipients=["admin@empresa.com"]
)
```

## Troubleshooting

### Problemas Comuns

1. **Métricas não sendo coletadas**
   - Verificar se os serviços estão iniciados
   - Conferir configuração do banco de dados
   - Validar permissões de escrita

2. **Alertas não funcionando**
   - Verificar configuração SMTP/webhook
   - Conferir regras de alerta
   - Validar thresholds configurados

3. **Dashboard não carregando**
   - Verificar se Flask está rodando
   - Conferir porta e endereço
   - Validar templates e arquivos estáticos

4. **API retornando erros**
   - Verificar logs estruturados
   - Conferir parâmetros da query
   - Validar conexão com banco de dados

### Logs de Debug

```python
# Habilitar logs detalhados
import logging
logging.basicConfig(level=logging.DEBUG)

# Verificar status dos serviços
from api.metrics_service import metrics_service
print(f"Metrics Service Status: {metrics_service.is_running()}")

from api.accuracy_tracker import accuracy_tracker
print(f"Accuracy Tracker Status: {accuracy_tracker.is_running()}")
```

## Manutenção

### Backup de Dados

```python
# Backup automático das métricas
from api.metrics_service import metrics_service

metrics_service.backup_data(
    backup_path="/backup/metrics",
    compress=True,
    retention_days=30
)
```

### Limpeza de Dados Antigos

```python
# Limpar dados com mais de 90 dias
metrics_service.cleanup_old_data(days=90)
accuracy_tracker.cleanup_old_data(days=90)
```

### Atualizações

1. **Atualizar Modelos**
```python
# Atualizar versão do modelo
metrics_service.update_model_version("v2.2")
```

2. **Migração de Dados**
```python
# Migrar dados para nova estrutura
from api.migrations import migrate_metrics_data
migrate_metrics_data(from_version="v1.0", to_version="v2.0")
```

## Performance e Otimização

### Configurações Recomendadas

```python
# Configurações de performance
METRICS_BATCH_SIZE = 1000  # Processar métricas em lotes
METRICS_RETENTION_DAYS = 365  # Manter dados por 1 ano
ALERT_CHECK_INTERVAL = 60  # Verificar alertas a cada minuto
LOG_ROTATION_SIZE = "100MB"  # Rotacionar logs a cada 100MB
```

### Monitoramento de Recursos

```python
# Monitorar uso de recursos
from api.structured_logger import structured_logger

# Log automático de recursos
structured_logger.log_system_resources()
```

## Integração com Outros Sistemas

### Webhook para Sistemas Externos

```python
# Configurar webhook para sistema externo
from api.alert_system import alert_system

alert_system.add_webhook(
    url="https://api.sistema-externo.com/alerts",
    headers={"Authorization": "Bearer token123"},
    events=["accuracy_drop", "system_error"]
)
```

### API para Integração

```python
# Exemplo de integração via API
import requests

# Obter métricas de outro sistema
response = requests.get(
    "http://localhost:5000/api/metrics/summary",
    headers={"Authorization": "Bearer api_token"}
)

metrics = response.json()
print(f"Acurácia atual: {metrics['data']['accuracy']['current']}")
```

## Conclusão

O Sistema de Monitoramento e Métricas fornece uma solução completa para acompanhar a performance do sistema Lotofácil, oferecendo:

- **Monitoramento em tempo real** de todas as métricas importantes
- **Alertas automáticos** para problemas de performance
- **Dashboard intuitivo** para visualização de dados
- **API robusta** para integração com outros sistemas
- **Logs estruturados** para análise detalhada
- **Retreinamento automático** baseado em performance

Para suporte adicional ou dúvidas, consulte os logs do sistema ou entre em contato com a equipe de desenvolvimento.

---

**Versão da Documentação:** 1.0  
**Última Atualização:** Janeiro 2024  
**Autor:** Sistema de IA - SOLO Coding