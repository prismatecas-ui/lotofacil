# Documentação Completa do Sistema - Lotofácil IA

## Visão Geral do Sistema

### Descrição
Sistema completo de previsão para Lotofácil utilizando Machine Learning com múltiplos modelos otimizados, API REST, interface web e sistema de monitoramento em tempo real.

### Versão Atual
- **Versão**: 2.0
- **Modelo Principal**: `modelo_final_extremo_20250919_113536.pkl`
- **Acurácia**: 81.09%
- **Última Atualização**: Setembro 2025

## Arquitetura do Sistema

### Componentes Principais

1. **Modelos de IA** (`/modelos/`)
2. **API REST** (`/api/`)
3. **Interface Web** (`/interface/`)
4. **Sistema de Monitoramento** (`/experimentos/`)
5. **Base de Dados** (`/dados/`, `/database/`)
6. **Cache Inteligente** (`/cache/`)

### Fluxo de Dados
```
Dados Históricos → Processamento → Treinamento → Modelo → API → Interface Web
                                      ↓
                              Cache ← Previsões
```

## Modelos de IA e Performance

### Modelos Disponíveis

#### 1. Modelo Final Extremo (Principal)
- **Arquivo**: `modelos/modelo_final_extremo_20250919_113536.pkl`
- **Acurácia**: 81.09%
- **Precisão**: 66.60%
- **Recall**: 81.09%
- **F1-Score**: 73.13%
- **ROC-AUC**: 61.77%
- **Técnicas**: Feature Engineering Extrema, SMOTETomek, Grid Search
- **Algoritmos**: Random Forest, Gradient Boosting, Extra Trees

#### 2. Ensemble Otimizado
- **Arquivo**: `modelos/ensemble_otimizado_20250919_103106.pkl`
- **Tipo**: Ensemble Voting Classifier
- **Componentes**: RF + GB + ET
- **Status**: Modelo secundário

#### 3. Modelo Super Otimizado
- **Arquivo**: `modelos/modelo_super_otimizado_20250919_103719.pkl`
- **Tipo**: Versão anterior otimizada
- **Status**: Backup/Comparação

### Resultados de Performance
- **Arquivo de Métricas**: `experimentos/resultados/modelo_final_extremo_20250919_113536.json`
- **Meta 80% Atingida**: ✅ Sim
- **Meta 85-90%**: ❌ Não (em desenvolvimento)
- **Features Utilizadas**: 49 features selecionadas
- **Amostras de Treino**: 2.792
- **Amostras de Teste**: 698

## APIs e Endpoints

### Servidor Principal
- **Arquivo**: `api/iniciar_api.py`
- **Porta**: 5000
- **URL Base**: `http://localhost:5000`

### Endpoints Disponíveis

#### Previsão
- **POST** `/predict`
- **Parâmetros**: `{"modelo": "string", "numeros": []}`
- **Resposta**: Números previstos + confiança

#### Status e Saúde
- **GET** `/health` - Status da API
- **GET** `/models` - Modelos disponíveis
- **GET** `/metrics` - Métricas do sistema

#### Cache
- **GET** `/cache/status` - Status do cache
- **POST** `/cache/clear` - Limpar cache

#### Monitoramento
- **GET** `/monitoring/stats` - Estatísticas em tempo real
- **GET** `/monitoring/logs` - Logs do sistema

### Configurações da API
- **Arquivo Principal**: `api/config.py`
- **Configurações JSON**: `api/config_api.json`
- **Requirements**: `api/requirements_api.txt`

## Interface Web

### Arquivos Principais
- **HTML**: `interface/index.html`
- **CSS**: `interface/styles.css`
- **JavaScript**: `interface/app.js`
- **README**: `interface/README.md`

### Funcionalidades
- Seleção de modelos
- Entrada manual de números
- Visualização de resultados
- Estatísticas e gráficos
- Modo claro/escuro
- Responsivo (mobile-friendly)

### Servidor Web
- **Comando**: `python -m http.server 8000 --directory interface`
- **URL**: `http://localhost:8000`

## Sistema de Monitoramento

### Scripts de Monitoramento

#### Monitor Simples
- **Arquivo**: `experimentos/monitor_simples.py`
- **Função**: Monitoramento em tempo real
- **Atualização**: A cada 30 segundos
- **Informações**:
  - Processos Python ativos
  - Arquivos recentes modificados
  - Progresso de otimizações
  - Tempo de execução
  - Status do sistema

#### Monitor de Otimização
- **Arquivo**: `experimentos/monitor_otimizacao.py`
- **Função**: Acompanhar treinamentos longos
- **Uso**: Durante otimizações de modelos

### Logs do Sistema
- **Diretório**: `logs/`
- **Arquivos Recentes**:
  - `logs/relatorio_startup_*.txt`
  - `logs/archive/` (logs arquivados)
  - `logs/lotofacil_completo_*/` (logs detalhados)

## Base de Dados

### Estrutura de Dados

#### Dados Principais
- **CSV Processado**: `dados/dados_processados.csv`
- **Base Original**: `base/resultados.csv`
- **SQLite**: `dados/lotofacil.db`

#### Backups
- **Diretório**: `base/`
- **Formato**: `backup_base_dados_YYYYMMDD_HHMMSS.xlsx`
- **Mais Recente**: `base/backup_base_dados_20250919_063916.xlsx`

### Scripts de Dados
- **Busca**: `dados/busca.py`
- **Processamento**: `dados/dados.py`
- **Scraping**: `dados/scrapping_resultados.py`
- **Combinações**: `dados/gerar_combinacoes.py`

## Cache Inteligente

### Configuração
- **Diretório**: `cache/`
- **Banco**: `cache/cache.db`
- **Arquivos**: `cache/*.cache`

### Funcionalidades
- Cache de previsões
- Cache de modelos carregados
- Invalidação automática
- Compressão de dados

## Experimentos e Desenvolvimento

### Diretório Principal
- **Localização**: `experimentos/`

### Scripts Importantes

#### Treinamento
- `experimentos/modelo_final_otimizado.py` - Treinamento principal
- `experimentos/treinar_modelo_completo.py` - Treinamento completo
- `experimentos/otimizar_modelo_avancado.py` - Otimizações avançadas

#### Análise
- `experimentos/exploratory_analysis.py` - Análise exploratória
- `experimentos/feature_engineering.py` - Engenharia de features
- `experimentos/diagnostico_sistema_completo.py` - Diagnóstico

#### Testes
- `experimentos/teste_performance_real.py` - Teste de performance
- `experimentos/testar_modelo_existente.py` - Teste de modelos
- `experimentos/teste_sistema_otimizado.py` - Teste do sistema

### Resultados
- **Diretório**: `experimentos/resultados/`
- **Relatórios**: `*.json`, `*.md`
- **Gráficos**: `*.png`

## Troubleshooting

### Problemas Comuns

#### API não responde
1. Verificar se está rodando: `http://localhost:5000/health`
2. Reiniciar: `python api/iniciar_api.py`
3. Verificar logs no terminal
4. Verificar se o modelo está carregado

#### Interface não carrega
1. Verificar servidor web: `http://localhost:8000`
2. Reiniciar: `python -m http.server 8000 --directory interface`
3. Limpar cache do navegador
4. Verificar console do navegador (F12)

#### Modelo não encontrado
1. Verificar arquivos em `modelos/`
2. Verificar configuração em `api/config.py`
3. Recarregar modelo via API

#### Performance baixa
1. Verificar uso de CPU/RAM
2. Limpar cache: `GET /cache/clear`
3. Reiniciar sistema completo
4. Verificar logs de erro

### Comandos de Diagnóstico

```bash
# Verificar status geral
python experimentos/diagnostico_sistema_completo.py

# Monitorar em tempo real
python experimentos/monitor_simples.py

# Testar performance
python experimentos/teste_performance_real.py

# Validar dados
python experimentos/validar_dados_cache.py
```

## Caminhos de Arquivos Importantes

### Configuração
```
api/config.py                    # Configurações principais da API
api/config_api.json             # Configurações JSON
requirements.txt                # Dependências Python
```

### Modelos
```
modelos/modelo_final_extremo_20250919_113536.pkl    # Modelo principal
modelos/ensemble_otimizado_20250919_103106.pkl      # Ensemble
modelos/modelo_super_otimizado_20250919_103719.pkl  # Backup
```

### Dados
```
dados/dados_processados.csv     # Dados processados
dados/lotofacil.db             # Base SQLite
base/resultados.csv            # Dados originais
```

### Interface
```
interface/index.html           # Página principal
interface/app.js              # Lógica JavaScript
interface/styles.css          # Estilos CSS
```

### Monitoramento
```
experimentos/monitor_simples.py        # Monitor principal
experimentos/monitor_otimizacao.py     # Monitor de treino
logs/                                  # Logs do sistema
```

### Resultados
```
experimentos/resultados/modelo_final_extremo_20250919_113536.json  # Métricas
resultados/                                                        # Previsões
cache/                                                            # Cache
```

## Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- pip ou conda
- Navegador moderno

### Instalação
```bash
# 1. Clonar repositório
git clone <repositorio>
cd lotofacil

# 2. Instalar dependências
pip install -r requirements.txt
pip install -r api/requirements_api.txt

# 3. Configurar base de dados
python dados/dados.py

# 4. Iniciar sistema
python api/iniciar_api.py &
python -m http.server 8000 --directory interface &
```

### Verificação
```bash
# Testar API
curl http://localhost:5000/health

# Testar Interface
open http://localhost:8000

# Monitorar sistema
python experimentos/monitor_simples.py
```

## Próximos Passos

### Melhorias Planejadas
1. **Acurácia**: Atingir meta de 85-90%
2. **Features**: Implementar padrões temporais avançados
3. **Interface**: Adicionar mais visualizações
4. **API**: Implementar autenticação
5. **Mobile**: App nativo

### Otimizações em Andamento
- Feature engineering avançada
- Ensemble com mais algoritmos
- Hyperparameter tuning automático
- Validação temporal aprimorada

---

**Documentação gerada em**: Setembro 2025  
**Versão do Sistema**: 2.0  
**Modelo Ativo**: modelo_final_extremo_20250919_113536.pkl (81.09%)  
**Contato**: Sistema Lotofácil IA