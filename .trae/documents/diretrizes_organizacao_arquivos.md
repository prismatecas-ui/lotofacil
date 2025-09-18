# Diretrizes de Organização de Arquivos - Projeto Lotofácil

## 1. Estrutura de Diretórios e Regras de Organização

### 📁 Regra Fundamental
**SEMPRE criar arquivos na pasta de referência apropriada conforme o tipo e propósito do arquivo.**

## 2. Mapeamento por Tipo de Arquivo

### 🗂️ Scripts (.py)
| Tipo de Script | Diretório | Exemplo |
|----------------|-----------|----------|
| **Scripts de execução** | `scripts/` | `update_data.py`, `backup_system.py` |
| **Scripts de experimento** | `experimentos/` | `test_model.py`, `analyze_data.py` |
| **Scripts de API** | `api/` | `caixa_integration.py`, `cache_manager.py` |
| **Scripts de modelo** | `modelo/` | `train_model.py`, `evaluate_model.py` |
| **Scripts de dados** | `dados/` | `process_data.py`, `validate_data.py` |
| **Scripts de análise** | `analises/` | `statistical_analysis.py` |
| **Scripts de processamento** | `processamento/` | `data_transformation.py` |
| **Scripts de funcionalidades** | `funcionalidades/` | `dashboard.py`, `reports.py` |

### 📊 Logs e Registros
| Tipo de Log | Diretório | Exemplo |
|-------------|-----------|----------|
| **Logs gerais** | `logs/` | `system.log`, `error.log` |
| **Logs de experimento** | `experimentos/logs/` | `experiment_20240918.log` |
| **Logs de execução** | Raiz do projeto | `log_execucao.txt` |
| **Logs de API** | `api/logs/` | `api_requests.log` |
| **Logs arquivados** | `logs/archive/` | `old_logs_2024.zip` |

### 📄 Dados e Bases
| Tipo de Dado | Diretório | Exemplo |
|--------------|-----------|----------|
| **Dados principais** | `base/` | `base_dados.xlsx`, `cache_concursos.json` |
| **Backups de dados** | `backup/` ou `backups/` | `backup_20240918.zip` |
| **Dados processados** | `experimentos/dados_processados/` | `features_engineered.csv` |
| **Datasets gerados** | `experimentos/datasets/` | `dataset_completo_20240918.csv` |
| **Dados brutos** | `data/raw/` | `raw_results.json` |
| **Dados de treinamento** | `data/training/` | `train_set.csv` |
| **Cache local** | `cache/` | `cache.db`, `*.cache` |

### 📈 Resultados e Saídas
| Tipo de Resultado | Diretório | Exemplo |
|-------------------|-----------|----------|
| **Resultados de experimentos** | `experimentos/resultados/` | `analysis_results.png` |
| **Resultados de predições** | `resultados/` | `ultimo_jogo.json` |
| **Gráficos e visualizações** | `experimentos/resultados/` | `correlation_plot.png` |
| **Relatórios** | `experimentos/resultados/` | `relatorio_fase1.md` |

### 🏗️ Modelos e Configurações
| Tipo de Arquivo | Diretório | Exemplo |
|------------------|-----------|----------|
| **Modelos de IA** | `modelo/` | `modelo_tensorflow2.py` |
| **Modelos treinados** | `experimentos/modelos/` | `model_v1.h5` |
| **Configurações** | `config/` | `config.json`, `settings.py` |
| **Preprocessadores** | `experimentos/modelos/` | `preprocessadores/` |

### 📚 Documentação
| Tipo de Documento | Diretório | Exemplo |
|-------------------|-----------|----------|
| **Documentação técnica** | `.trae/documents/` | `arquitetura_tecnica.md` |
| **Documentação geral** | `docs/` | `README_ACESSO.md` |
| **Análises documentadas** | `docs/` | `analise_base_dados.md` |

### 🧪 Testes
| Tipo de Teste | Diretório | Exemplo |
|---------------|-----------|----------|
| **Testes unitários** | `tests/` | `test_model.py` |
| **Testes de integração** | `tests/` | `test_api_integration.py` |
| **Validação** | `validation/` | `validate_results.py` |

## 3. Regras Específicas por Contexto

### 🎯 Ao Criar Scripts de Experimento
```
✅ CORRETO: experimentos/novo_experimento.py
❌ ERRADO: novo_experimento.py (na raiz)
❌ ERRADO: scripts/novo_experimento.py
```

### 📊 Ao Gerar Logs
```
✅ CORRETO: logs/sistema_20240918.log
✅ CORRETO: experimentos/logs/experimento_20240918.log
❌ ERRADO: experimento_20240918.log (na raiz)
```

### 💾 Ao Salvar Dados
```
✅ CORRETO: experimentos/datasets/dataset_completo.csv
✅ CORRETO: base/cache_concursos.json
❌ ERRADO: dataset_completo.csv (na raiz)
```

### 📈 Ao Gerar Resultados
```
✅ CORRETO: experimentos/resultados/analise_grafico.png
✅ CORRETO: resultados/predicao_final.json
❌ ERRADO: analise_grafico.png (na raiz)
```

## 4. Checklist de Criação de Arquivos

### ✅ Antes de Criar Qualquer Arquivo
1. **Identificar o tipo** do arquivo (script, log, dado, resultado, etc.)
2. **Consultar a tabela** de mapeamento acima
3. **Verificar se o diretório existe**, criar se necessário
4. **Usar nomenclatura consistente** com padrão do projeto
5. **Adicionar timestamp** quando apropriado

### ✅ Nomenclatura Padrão
```
# Scripts
nome_funcionalidade.py
processar_dados_lotofacil.py

# Logs
log_tipo_YYYYMMDD_HHMMSS.log
log_experimento_20240918_143022.log

# Dados/Datasets
dataset_nome_YYYYMMDD_HHMMSS.csv
dataset_completo_20240918_143022.csv

# Resultados
resultado_tipo_YYYYMMDD_HHMMSS.ext
analise_correlacao_20240918_143022.png

# Modelos
modelo_versao_YYYYMMDD.h5
modelo_v2_20240918.h5
```

## 5. Estrutura de Diretórios Recomendada

```
lotofacil/
├── 📁 scripts/           # Scripts de execução e automação
├── 📁 experimentos/       # Scripts de experimento e análise
│   ├── 📁 logs/          # Logs específicos de experimentos
│   ├── 📁 datasets/      # Datasets gerados
│   ├── 📁 resultados/    # Resultados de análises
│   ├── 📁 modelos/       # Modelos treinados
│   └── 📁 dados_processados/ # Dados intermediários
├── 📁 logs/              # Logs gerais do sistema
│   └── 📁 archive/       # Logs arquivados
├── 📁 base/              # Dados principais
├── 📁 backup/            # Backups
├── 📁 cache/             # Cache local
├── 📁 data/              # Dados organizados
│   ├── 📁 raw/          # Dados brutos
│   └── 📁 training/     # Dados de treinamento
├── 📁 resultados/        # Resultados de predições
├── 📁 modelo/            # Código dos modelos
├── 📁 api/               # Scripts de API
├── 📁 dados/             # Scripts de manipulação de dados
├── 📁 analises/          # Scripts de análise
├── 📁 processamento/     # Scripts de processamento
├── 📁 funcionalidades/   # Scripts de funcionalidades
├── 📁 tests/             # Testes
├── 📁 docs/              # Documentação geral
├── 📁 .trae/documents/   # Documentação técnica
└── 📁 config/            # Configurações
```

## 6. Comandos de Criação Automática

### 🚀 Template para Novos Scripts
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nome do Script: [nome_do_script].py
Propósito: [Descrever o propósito]
Diretório: [diretório_apropriado]/
Criado em: [data]
Autor: SOLO Document Agent
"""

import os
import sys
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('[diretório_logs]/[nome_script]_YYYYMMDD.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Função principal"""
    logger.info("Iniciando [nome_do_script]...")
    # Código aqui
    logger.info("[nome_do_script] concluído com sucesso!")

if __name__ == "__main__":
    main()
```

### 🗂️ Criação Automática de Diretórios
```python
# Template para criar estrutura de diretórios
import os

def criar_estrutura_diretorios():
    """Cria estrutura padrão de diretórios"""
    diretorios = [
        'scripts',
        'experimentos/logs',
        'experimentos/datasets', 
        'experimentos/resultados',
        'experimentos/modelos',
        'experimentos/dados_processados',
        'logs/archive',
        'data/raw',
        'data/training',
        'config'
    ]
    
    for diretorio in diretorios:
        os.makedirs(diretorio, exist_ok=True)
        print(f"✅ Diretório criado: {diretorio}")
```

## 7. Validação e Auditoria

### 🔍 Script de Validação da Estrutura
```python
# Salvar como: scripts/validar_estrutura_arquivos.py

def validar_estrutura():
    """Valida se arquivos estão nos diretórios corretos"""
    problemas = []
    
    # Verificar scripts na raiz
    for arquivo in os.listdir('.'):
        if arquivo.endswith('.py') and arquivo not in ['jogar.py', 'check_tables.py']:
            problemas.append(f"Script na raiz: {arquivo} - Mover para scripts/ ou experimentos/")
    
    # Verificar logs na raiz
    for arquivo in os.listdir('.'):
        if arquivo.startswith('log_') or arquivo.endswith('.log'):
            problemas.append(f"Log na raiz: {arquivo} - Mover para logs/")
    
    return problemas
```

## 8. Integração com Documentação

### 📝 Atualização Automática da Documentação
Sempre que criar um arquivo, adicionar entrada em:
- `registro_completo_arquivos_projeto.md`
- `indice_consulta_rapida_projeto.md`

### 🔄 Template de Atualização
```markdown
## [Fase/Seção Apropriada]

### [Subsecção]
- **[nome_arquivo]**
  - Propósito: [descrição]
  - Localização: `[diretório]/[nome_arquivo]`
  - Dependências: [lista]
  - Criado em: [data]
  - Status: [ativo/deprecated/em desenvolvimento]
```

---

**📋 Resumo das Regras Principais:**
1. **Scripts** → `scripts/`, `experimentos/`, `api/`, `modelo/`, etc.
2. **Logs** → `logs/`, `experimentos/logs/`
3. **Dados** → `base/`, `experimentos/datasets/`, `data/`
4. **Resultados** → `experimentos/resultados/`, `resultados/`
5. **Documentação** → `.trae/documents/`, `docs/`

**🔄 Última Atualização**: 18/09/2024  
**📋 Versão**: 1.0  
**👤 Responsável**: SOLO Document Agent

> **⚠️ IMPORTANTE**: Seguir estas diretrizes é OBRIGATÓRIO para manter a organização e facilitar a manutenção do projeto!