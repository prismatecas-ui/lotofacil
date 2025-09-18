# Índice de Consulta Rápida - Projeto Lotofácil

## 1. Consulta por Problema/Necessidade

### 🔍 "Preciso analisar dados históricos"
- **Arquivo**: `experimentos/exploratory_analysis.py`
- **Comando**: `python experimentos/exploratory_analysis.py`
- **Saída**: Gráficos em `experimentos/resultados/`
- **Dependências**: pandas, matplotlib, seaborn

### 🎯 "Preciso gerar predições"
- **Arquivo**: `jogar.py`
- **Comando**: `python jogar.py`
- **Pré-requisito**: TensorFlow instalado
- **Cache**: `base/cache_concursos.json`

### 📊 "Preciso criar features avançadas"
- **Arquivo**: `experimentos/feature_engineering.py`
- **Classes**: `FeatureEngineeringLotofacil`
- **Métodos**: `criar_features_estatisticas()`, `criar_features_temporais()`

### 🗄️ "Preciso gerar dataset completo"
- **Arquivo**: `experimentos/gerar_dataset_completo.py`
- **Comando**: `python experimentos/gerar_dataset_completo.py`
- **⚠️ PROBLEMA CONHECIDO**: Limitado a 1991 concursos
- **Debug**: `experimentos/debug_dados.py`

### 🔄 "Preciso atualizar dados da Caixa"
- **Arquivo**: `api/caixa_api.py`
- **Cache**: `base/cache_concursos.json`
- **Auto-update**: `api/auto_update.py`

### 📈 "Preciso métricas do modelo"
- **Arquivo**: `experimentos/advanced_metrics.py`
- **Análise**: `experimentos/model_limitations_analyzer.py`
- **Relatório**: `experimentos/relatorio_completo_fase1.py`

## 2. Consulta por Tipo de Arquivo

### 📋 Scripts de Execução Direta
```bash
# Sistema principal
python jogar.py

# Análise exploratória
python experimentos/exploratory_analysis.py

# Geração de dataset
python experimentos/gerar_dataset_completo.py

# Relatório completo
python experimentos/relatorio_completo_fase1.py

# Debug de dados
python experimentos/debug_dados.py
```

### 🏗️ Módulos/Classes Importáveis
```python
# Feature Engineering
from experimentos.feature_engineering import FeatureEngineeringLotofacil

# Seleção de Features
from experimentos.feature_selector import FeatureSelectorLotofacil

# Preprocessamento
from experimentos.preprocessor import PreprocessadorAvancadoLotofacil

# Dados
from dados.dados import carregar_dados, preparar_dados

# Modelo IA
from modelo.modelo_tensorflow2 import LotofacilModel
```

### 📁 Arquivos de Dados Críticos
- `base/cache_concursos.json` - **CRÍTICO** - Cache principal
- `base/base_dados.xlsx` - Dados históricos
- `dados/lotofacil.db` - Banco SQLite
- `cache/cache.db` - Cache local

## 3. Fluxo de Trabalho Recomendado

### 🚀 Para Iniciar Novo Desenvolvimento
1. **Consultar**: `registro_completo_arquivos_projeto.md`
2. **Verificar**: Se funcionalidade já existe
3. **Escolher**: Diretório apropriado (`experimentos/`, `api/`, `modelo/`)
4. **Criar**: Arquivo seguindo padrões existentes
5. **Atualizar**: Documentação após criação

### 🔧 Para Debug/Correção
1. **Identificar**: Problema na seção "Problemas Pendentes"
2. **Localizar**: Arquivo responsável
3. **Usar**: Script de debug correspondente
4. **Documentar**: Solução encontrada

### 📊 Para Análise de Dados
1. **Verificar**: `base/cache_concursos.json` atualizado
2. **Executar**: `experimentos/exploratory_analysis.py`
3. **Gerar**: Features com `feature_engineering.py`
4. **Criar**: Dataset com `gerar_dataset_completo.py`
5. **Analisar**: Resultados em `experimentos/resultados/`

## 4. Checklist de Verificação Rápida

### ✅ Antes de Criar Arquivo
- [ ] Consultei `registro_completo_arquivos_projeto.md`?
- [ ] Verifiquei se funcionalidade já existe?
- [ ] Escolhi diretório correto?
- [ ] Segui padrão de nomenclatura?

### ✅ Antes de Executar Script
- [ ] Ambiente virtual ativado?
- [ ] Dependências instaladas?
- [ ] Dados atualizados?
- [ ] Espaço em disco suficiente?

### ✅ Após Modificação
- [ ] Testei a funcionalidade?
- [ ] Atualizei documentação?
- [ ] Commitei mudanças?
- [ ] Documentei decisões técnicas?

## 5. Comandos de Emergência

### 🚨 Sistema Não Funciona
```bash
# Verificar ambiente
python --version
pip list | grep tensorflow

# Reativar ambiente
cd venv_lotofacil
Scripts\activate

# Reinstalar dependências
pip install -r requirements.txt
```

### 🚨 Dados Corrompidos
```bash
# Backup disponível
cp base/backup_base_dados_*.xlsx base/base_dados.xlsx

# Recriar cache
python api/caixa_api.py
```

### 🚨 Modelo Não Treina
```bash
# Debug do modelo
python experimentos/model_limitations_analyzer.py

# Verificar dados
python experimentos/debug_dados.py
```

## 6. Contatos por Funcionalidade

| Funcionalidade | Arquivo Principal | Script Debug | Documentação |
|----------------|-------------------|--------------|-------------|
| **Cache Sistema** | `base/cache_concursos.json` | `api/caixa_api.py` | Seção 2 do registro |
| **Predição IA** | `jogar.py` | `experimentos/model_limitations_analyzer.py` | Seção 4.1 |
| **Dataset** | `experimentos/gerar_dataset_completo.py` | `experimentos/debug_dados.py` | Seção 3.5 |
| **Features** | `experimentos/feature_engineering.py` | `experimentos/exploratory_analysis.py` | Seção 3.4 |
| **API Caixa** | `api/caixa_api.py` | `api/auto_update.py` | Seção 4.3 |

## 7. Atalhos de Desenvolvimento

### 🔥 Desenvolvimento Rápido
```python
# Template básico para novo script
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos locais
from dados.dados import carregar_dados
from experimentos.experiment_logger import ExperimentLogger
```

### 🎯 Análise Rápida
```python
# Carregar dados rapidamente
from dados.dados import carregar_dados
dados = carregar_dados('Importar_Ciclo')
print(f"Dados carregados: {len(dados)} registros")
```

### 📊 Visualização Rápida
```python
# Plot básico
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
# ... código do plot
plt.savefig('experimentos/resultados/plot_temp.png')
```

## 8. Troubleshooting Comum

### ❌ "ModuleNotFoundError"
- **Solução**: Ativar ambiente virtual
- **Comando**: `venv_lotofacil\Scripts\activate`

### ❌ "FileNotFoundError: base_dados.xlsx"
- **Solução**: Verificar arquivo em `base/`
- **Backup**: Usar `backup_base_dados_*.xlsx`

### ❌ "Apenas 1991 concursos carregados"
- **Status**: **PROBLEMA CONHECIDO**
- **Debug**: `experimentos/debug_dados.py`
- **Causa**: Limitação em `dados/dados.py`

### ❌ "TensorFlow não encontrado"
- **Solução**: `pip install tensorflow`
- **Verificar**: `python -c "import tensorflow; print(tensorflow.__version__)"`

## 9. Estrutura de Logs e Saídas

### 📝 Logs do Sistema
- `logs/` - Logs gerais
- `experimentos/logs/` - Logs de experimentos
- `log_execucao*.txt` - Logs de execução

### 📊 Resultados e Saídas
- `experimentos/resultados/` - Gráficos e análises
- `experimentos/datasets/` - Datasets gerados
- `resultados/` - Resultados de predições

---

**🔄 Última Atualização**: 18/09/2024  
**📋 Versão**: 1.0  
**👤 Responsável**: SOLO Document Agent

> **💡 DICA**: Mantenha este documento aberto durante desenvolvimento para consulta rápida!