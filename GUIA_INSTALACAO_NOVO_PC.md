# Guia de Instalação - Sistema Lotofácil em Novo PC

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Git
- Conexão com internet (para download do modelo)

## 🚀 Instalação Rápida

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/lotofacil.git
cd lotofacil
```

### 2. Criar Ambiente Virtual
```bash
# Windows
python -m venv venv_lotofacil
venv_lotofacil\Scripts\activate

# Linux/Mac
python3 -m venv venv_lotofacil
source venv_lotofacil/bin/activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar Modelo de IA
```bash
# Configuração automática (recomendado)
python scripts/gerenciar_modelo.py configurar

# OU verificar se modelo existe
python scripts/gerenciar_modelo.py verificar
```

### 5. Inicializar Sistema
```bash
# Terminal 1: Iniciar API
python api/iniciar_api.py

# Terminal 2: Iniciar Interface Web
python -m http.server 8000 --directory interface
```

### 6. Acessar Sistema
Abra o navegador em: `http://localhost:8000`

## 🔧 Configuração Detalhada

### Gerenciamento do Modelo de IA

O sistema inclui um script para gerenciar automaticamente o modelo:

```bash
# Verificar se modelo existe e está íntegro
python scripts/gerenciar_modelo.py verificar

# Baixar modelo do repositório remoto
python scripts/gerenciar_modelo.py baixar

# Criar backup do modelo atual
python scripts/gerenciar_modelo.py backup

# Restaurar modelo do backup
python scripts/gerenciar_modelo.py restaurar

# Configuração automática (tenta todas as opções)
python scripts/gerenciar_modelo.py configurar
```

### Estrutura de Pastas Importantes

```
lotofacil/
├── api/                    # API do sistema
├── interface/              # Interface web
├── scripts/                # Scripts utilitários
│   └── gerenciar_modelo.py # Gerenciador do modelo IA
├── modelos/                # Modelos de IA (ignorado no Git)
├── backup/                 # Backups locais
├── requirements.txt        # Dependências Python
└── GUIA_INSTALACAO_NOVO_PC.md
```

## 🎯 Solução de Problemas

### Modelo não encontrado
```bash
# Tentar configuração automática
python scripts/gerenciar_modelo.py configurar

# Se falhar, verificar conexão e tentar download manual
python scripts/gerenciar_modelo.py baixar
```

### Erro de dependências
```bash
# Atualizar pip
python -m pip install --upgrade pip

# Reinstalar dependências
pip install -r requirements.txt --force-reinstall
```

### Porta já em uso
```bash
# Usar porta diferente para interface
python -m http.server 8001 --directory interface

# Ou para API (editar api/iniciar_api.py)
```

## 📊 Otimização de Espaço

### Arquivos que NÃO são necessários para funcionamento:
- `venv_lotofacil/` (2GB) - Recriar com `pip install -r requirements.txt`
- `combinacoes/` (150MB) - Será regenerado automaticamente
- `logs/*.log` - Arquivos de log antigos
- `cache/` - Cache temporário
- `backup/` - Backups locais (opcional)

### Para reduzir tamanho:
```bash
# Limpar arquivos temporários
python -c "import os, shutil; [shutil.rmtree(d, ignore_errors=True) for d in ['cache', '__pycache__', '.pytest_cache'] if os.path.exists(d)]"

# Remover logs antigos
find . -name '*.log' -delete  # Linux/Mac
Get-ChildItem -Recurse -Filter '*.log' | Remove-Item  # Windows PowerShell
```

## 🔄 Backup e Migração

### Fazer backup completo
```bash
# Backup do modelo
python scripts/gerenciar_modelo.py backup

# Backup de configurações (se houver)
cp config.py config_backup.py
```

### Migrar para novo PC
1. Clonar repositório no novo PC
2. Instalar dependências
3. Executar configuração automática do modelo
4. Restaurar configurações personalizadas (se houver)

## 📝 Notas Importantes

- **Tamanho mínimo**: ~50MB (sem ambiente virtual)
- **Tamanho com venv**: ~2GB (incluindo todas as dependências)
- **Modelo de IA**: ~120MB (baixado automaticamente)
- **Dados de combinações**: ~150MB (gerados automaticamente)

## 🆘 Suporte

Se encontrar problemas:
1. Verificar logs em `logs/`
2. Executar `python scripts/gerenciar_modelo.py verificar`
3. Consultar este guia
4. Abrir issue no repositório GitHub

---

**Tempo estimado de instalação**: 5-10 minutos (dependendo da velocidade da internet)