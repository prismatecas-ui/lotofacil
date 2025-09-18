# Registro Completo de Arquivos e Componentes - Projeto Lotofácil

## 1. Visão Geral do Projeto

Este documento mantém um registro detalhado de todos os arquivos, scripts e componentes criados durante o desenvolvimento do sistema de predição da Lotofácil, organizados por fases e tarefas específicas.

## 2. Sistema de Cache e Integração com Banco de Dados

### 2.1 Arquivo Principal: cache_concursos.json
- **Localização**: `./base/cache_concursos.json`
- **Propósito**: Alimentar o banco de dados com resultados dos concursos
- **Formato**: JSON com estrutura de concursos da Lotofácil
- **Integração**: Conecta com `dados/dados.py` e `jogar.py`
- **Atualização**: Via API da Caixa através de `api/caixa_api.py`

### 2.2 Arquivos Relacionados ao Cache
- `base/base_dados.xlsx` - Planilha principal com dados históricos
- `cache/cache.db` - Banco SQLite para cache local
- `cache/8781183c1b0012388f653e1a92404672.cache` - Cache específico

## 3. Registro por Fases do Projeto

### FASE 1: Análise Exploratória e Preparação do Ambiente

#### 3.1 Arquivos de Análise Exploratória
- **experimentos/exploratory_analysis.py**
  - Propósito: Análise estatística dos dados históricos
  - Dependências: pandas, numpy, matplotlib, seaborn
  - Saída: Gráficos e métricas em `experimentos/resultados/`

- **experimentos/experiment_logger.py**
  - Propósito: Sistema de logging para experimentos
  - Funcionalidades: Log estruturado, métricas de performance
  - Integração: Usado por todos os scripts de experimento

- **experimentos/model_limitations_analyzer.py**
  - Propósito: Análise das limitações do modelo atual
  - Saída: Relatório de limitações em `experimentos/resultados/`
  - Dependências: TensorFlow, scikit-learn

#### 3.2 Arquivos de Relatório da Fase 1
- **experimentos/relatorio_completo_fase1.py**
  - Propósito: Geração de relatório consolidado da Fase 1
  - Saída: `experimentos/resultados/relatorio_completo_fase1_YYYYMMDD_HHMMSS.md`
  - Integração: Consolida dados de todos os análises da Fase 1

#### 3.3 Resultados Gerados na Fase 1
- `experimentos/resultados/dados_analise_exploratoria_20250918_144540.json`
- `experimentos/resultados/dados_limitacoes_modelo_20250918_144845.json`
- `experimentos/resultados/relatorio_limitacoes_modelo_20250918_144845.md`
- `experimentos/resultados/relatorio_completo_fase1_20250918_145357.md`
- `experimentos/resultados/dashboard_resumo_fase1_20250918_145358.png`
- Gráficos: `correlation_analysis.png`, `distribution_analysis.png`, `frequency_analysis.png`

### FASE 2: Feature Engineering e Seleção de Features

#### 3.4 Arquivos de Feature Engineering
- **experimentos/feature_engineering.py**
  - Propósito: Criação de features avançadas (estatísticas, temporais, padrões)
  - Classes: `FeatureEngineeringLotofacil`
  - Métodos: `criar_features_estatisticas()`, `criar_features_temporais()`, `criar_features_padroes()`

- **experimentos/feature_selector.py**
  - Propósito: Seleção automática das melhores features
  - Classes: `FeatureSelectorLotofacil`
  - Métodos: `selecionar_melhores_features()`, `analisar_importancia()`

- **experimentos/preprocessor.py**
  - Propósito: Preprocessamento avançado dos dados
  - Classes: `PreprocessadorAvancadoLotofacil`
  - Funcionalidades: Normalização, tratamento de outliers, encoding

#### 3.5 Arquivos de Geração de Dataset
- **experimentos/gerar_dataset_completo.py**
  - Propósito: Pipeline completo para geração do dataset final
  - Classes: `GeradorDatasetCompleto`
  - Saída: Datasets em `experimentos/datasets/`
  - Status: **PROBLEMA IDENTIFICADO** - Processando apenas 1991 concursos

- **experimentos/debug_dados.py**
  - Propósito: Debug do problema de limitação de dados
  - Status: Em desenvolvimento

#### 3.6 Arquivos de Métricas Avançadas
- **experimentos/advanced_metrics.py**
  - Propósito: Métricas avançadas para avaliação do modelo
  - Funcionalidades: Precision, Recall, F1-Score, ROC-AUC

### FASE 3: Otimização de Modelos (Planejada)

#### 3.7 Documentação de Otimização
- **.trae/documents/guia_otimizacao_ia_lotofacil.md**
  - Propósito: Guia técnico completo para otimização da IA
  - Seções: 11 estratégias de otimização com código Python
  - Status: Documento criado, implementação em andamento

## 4. Arquivos do Sistema Principal

### 4.1 Sistema de Predição
- **jogar.py** - Script principal do sistema
  - Status: TensorFlow reativado
  - Funcionalidades: Geração de jogos inteligentes, treinamento de modelo
  - Dependências: TensorFlow, pandas, numpy, sqlite3

- **modelo/modelo_tensorflow2.py** - Modelo de IA principal
  - Classes: `LotofacilModel`
  - Funcionalidades: Rede neural avançada, callbacks, métricas

### 4.2 Sistema de Dados
- **dados/dados.py** - Módulo de manipulação de dados
  - Funções: `carregar_dados()`, `preparar_dados()`, `dividir_dados()`
  - **LIMITAÇÃO IDENTIFICADA**: Carrega apenas aba 'Importar_Ciclo'

- **dados/lotofacil.db** - Banco SQLite principal
- **database/lotofacil.db** - Backup do banco

### 4.3 Sistema de API
- **api/caixa_api.py** - Integração com API da Caixa
- **api/cache_service.py** - Serviço de cache
- **api/auto_update.py** - Atualização automática

## 5. Índice de Consulta Rápida

### 5.1 Por Funcionalidade

| Funcionalidade | Arquivo Principal | Arquivos Relacionados |
|----------------|-------------------|----------------------|
| Análise Exploratória | `experimentos/exploratory_analysis.py` | `experiment_logger.py` |
| Feature Engineering | `experimentos/feature_engineering.py` | `feature_selector.py`, `preprocessor.py` |
| Geração Dataset | `experimentos/gerar_dataset_completo.py` | `debug_dados.py` |
| Predição IA | `jogar.py` | `modelo/modelo_tensorflow2.py` |
| Cache Sistema | `base/cache_concursos.json` | `api/cache_service.py` |
| Dados Históricos | `base/base_dados.xlsx` | `dados/dados.py` |
| API Caixa | `api/caixa_api.py` | `api/auto_update.py` |
| Logging | `experimentos/experiment_logger.py` | Todos os scripts |
| Métricas | `experimentos/advanced_metrics.py` | `model_limitations_analyzer.py` |
| Relatórios | `experimentos/relatorio_completo_fase1.py` | Arquivos em `resultados/` |

### 5.2 Por Diretório

#### experimentos/
- `advanced_metrics.py` - Métricas avançadas
- `debug_dados.py` - Debug de dados
- `experiment_logger.py` - Sistema de logging
- `exploratory_analysis.py` - Análise exploratória
- `feature_engineering.py` - Engenharia de features
- `feature_selector.py` - Seleção de features
- `gerar_dataset_completo.py` - Geração de dataset
- `model_limitations_analyzer.py` - Análise de limitações
- `preprocessor.py` - Preprocessamento
- `relatorio_completo_fase1.py` - Relatório Fase 1

#### base/
- `base_dados.xlsx` - Dados históricos principais
- `cache_concursos.json` - **ARQUIVO CRÍTICO** - Cache de concursos
- `resultados.csv` - Resultados processados

#### api/
- `caixa_api.py` - API da Caixa
- `cache_service.py` - Serviço de cache
- `auto_update.py` - Atualização automática

## 6. Histórico de Implementações

### 6.1 Decisões Técnicas Importantes

#### TensorFlow vs Análise Estatística
- **Data**: Setembro 2024
- **Decisão**: Reativar TensorFlow após problemas de dependências
- **Justificativa**: Melhor precisão nas predições
- **Arquivos Afetados**: `jogar.py`, `modelo/modelo_tensorflow2.py`

#### Sistema de Cache
- **Data**: Implementação inicial
- **Decisão**: Usar `cache_concursos.json` como fonte principal
- **Justificativa**: Performance e redução de chamadas à API
- **Arquivos Criados**: `base/cache_concursos.json`, `api/cache_service.py`

#### Feature Engineering Avançado
- **Data**: Fase 2 do projeto
- **Decisão**: Implementar sistema modular de features
- **Justificativa**: Flexibilidade e manutenibilidade
- **Arquivos Criados**: `feature_engineering.py`, `feature_selector.py`, `preprocessor.py`

### 6.2 Problemas Identificados e Soluções

#### Problema: Dataset Limitado (1991 concursos)
- **Status**: **ATIVO**
- **Arquivo Afetado**: `experimentos/gerar_dataset_completo.py`
- **Causa Provável**: Limitação na função `carregar_dados()` em `dados/dados.py`
- **Solução Proposta**: Verificar aba 'Importar_Ciclo' e outras abas disponíveis
- **Arquivo de Debug**: `experimentos/debug_dados.py`

#### Problema: Dependências TensorFlow
- **Status**: **RESOLVIDO**
- **Solução**: Instalação completa no ambiente virtual
- **Arquivos Modificados**: `jogar.py`, `requirements.txt`

## 7. Guia de Manutenção

### 7.1 Checklist Antes de Criar Novos Arquivos

1. **Consultar este documento** para verificar se funcionalidade já existe
2. **Verificar diretório `experimentos/`** para scripts similares
3. **Checar `api/`** para integrações existentes
4. **Revisar `modelo/`** para funcionalidades de IA
5. **Consultar `dados/`** para manipulação de dados

### 7.2 Processo de Atualização da Documentação

1. **Após criar arquivo**: Adicionar entrada na seção correspondente
2. **Após modificar arquivo**: Atualizar histórico de implementações
3. **Após resolver problema**: Documentar solução na seção 6.2
4. **Mensalmente**: Revisar e consolidar documentação

### 7.3 Comandos de Execução Principais

```bash
# Executar sistema principal
python jogar.py

# Gerar dataset completo
python experimentos/gerar_dataset_completo.py

# Análise exploratória
python experimentos/exploratory_analysis.py

# Relatório Fase 1
python experimentos/relatorio_completo_fase1.py

# Debug de dados
python experimentos/debug_dados.py
```

### 7.4 Estrutura de Logs

- **Logs de Experimento**: `experimentos/logs/`
- **Logs do Sistema**: `logs/`
- **Logs de Execução**: `log_execucao*.txt`

## 8. Próximos Passos e TODOs

### 8.1 Problemas Pendentes
- [ ] **CRÍTICO**: Resolver limitação de 1991 concursos no dataset
- [ ] Implementar Fase 3 de otimização conforme `guia_otimizacao_ia_lotofacil.md`
- [ ] Validar integridade do `cache_concursos.json`
- [ ] Otimizar performance do `gerar_dataset_completo.py`

### 8.2 Melhorias Planejadas
- [ ] Sistema de versionamento automático de datasets
- [ ] Dashboard web para monitoramento
- [ ] API REST para predições
- [ ] Testes automatizados

## 9. Contatos e Responsabilidades

- **Sistema Principal**: `jogar.py`
- **Cache e Dados**: `base/cache_concursos.json`, `dados/dados.py`
- **Experimentos**: Diretório `experimentos/`
- **Documentação**: `.trae/documents/`

---

**Última Atualização**: 18/09/2024
**Versão do Documento**: 1.0
**Responsável**: SOLO Document Agent

> **IMPORTANTE**: Este documento deve ser consultado SEMPRE antes de criar novos arquivos ou implementar funcionalidades. Manter atualizado após cada modificação no projeto.