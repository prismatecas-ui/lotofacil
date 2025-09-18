# Sistema de Backup e Versionamento - Lotofácil

## Visão Geral

Este sistema fornece uma solução completa de backup e versionamento para modelos de machine learning do projeto Lotofácil. O sistema oferece funcionalidades avançadas de versionamento semântico, backup automático, validação de integridade e restauração de modelos.

## Características Principais

### 🔄 Versionamento Semântico
- Versionamento automático seguindo padrão semântico (major.minor.patch)
- Controle de versões baseado em tipos de mudança
- Histórico completo de todas as versões

### 💾 Backup Automático
- Agendamento flexível de backups
- Monitoramento de diretórios em tempo real
- Backup incremental e completo
- Compressão automática para economia de espaço

### 🔍 Validação e Integridade
- Verificação de integridade de arquivos
- Validação de hash para detectar corrupção
- Teste de carregamento de modelos
- Relatórios de validação detalhados

### 📊 Métricas e Relatórios
- Métricas específicas para Lotofácil
- Relatórios de performance e uso
- Análise de crescimento e tendências
- Dashboards de monitoramento

### 🔐 Segurança
- Criptografia opcional de backups
- Controle de acesso
- Assinatura digital de arquivos
- Logs de auditoria

## Estrutura de Arquivos

```
modelo/
├── backup_versionamento.py      # Classe principal do sistema
├── inicializar_backup.py         # Script de inicialização
├── config_backup.json           # Configuração do sistema
├── teste_sistema_backup.py      # Testes automatizados
├── requirements_backup.txt      # Dependências específicas
└── README_BACKUP.md            # Esta documentação
```

## Instalação

### 1. Instalar Dependências

```bash
pip install -r requirements_backup.txt
```

### 2. Configurar Sistema

Edite o arquivo `config_backup.json` conforme suas necessidades:

```json
{
  "diretorios": {
    "base": "./backups_modelos",
    "modelos": "./backups_modelos/modelos"
  },
  "backup_automatico": {
    "habilitado": true,
    "intervalo_horas": 6
  }
}
```

### 3. Inicializar Sistema

```bash
python inicializar_backup.py iniciar --config config_backup.json
```

## Uso Básico

### Inicializar Sistema

```python
from backup_versionamento import GerenciadorBackupVersionamento

# Criar gerenciador
gerenciador = GerenciadorBackupVersionamento("./backups")

# Criar backup
versao = gerenciador.criar_backup(
    "modelo.h5",
    tipo_mudanca="melhoria",
    descricao="Melhorias na arquitetura"
)

print(f"Backup criado: versão {versao}")
```

### Listar Versões

```python
# Listar todas as versões de um modelo
versoes = gerenciador.listar_versoes("modelo.h5")

for versao in versoes:
    print(f"Versão {versao['versao']}: {versao['descricao']}")
```

### Restaurar Backup

```python
# Restaurar versão específica
caminho_restaurado = gerenciador.restaurar_backup(
    "modelo.h5",
    "1.2.3",
    "./restauracao/"
)

print(f"Modelo restaurado em: {caminho_restaurado}")
```

## Uso via Linha de Comando

### Iniciar Sistema

```bash
# Iniciar em modo daemon
python inicializar_backup.py iniciar --daemon

# Iniciar com configuração específica
python inicializar_backup.py iniciar --config minha_config.json
```

### Criar Backup Manual

```bash
python inicializar_backup.py backup --modelo modelo.h5 --descricao "Backup manual"
```

### Verificar Status

```bash
python inicializar_backup.py status
```

### Ver Configuração

```bash
python inicializar_backup.py config
```

## Configuração Avançada

### Backup Automático

```json
{
  "backup_automatico": {
    "habilitado": true,
    "horarios_execucao": ["02:00", "08:00", "14:00", "20:00"],
    "diretorios_monitorados": ["./modelo", "./models"],
    "extensoes_monitoradas": [".h5", ".pkl", ".joblib"],
    "ignorar_arquivos": ["temp_*", "*.tmp"]
  }
}
```

### Políticas de Retenção

```json
{
  "retencao": {
    "por_tempo": {
      "ultimos_7_dias": "todas",
      "ultimos_30_dias": "diarias",
      "ultimos_90_dias": "semanais"
    },
    "por_quantidade": {
      "maximo_versoes_por_modelo": 50
    }
  }
}
```

### Tipos de Modelo Suportados

- **TensorFlow**: `.h5`, `.pb`, `.tflite`
- **Scikit-learn**: `.pkl`, `.joblib`
- **XGBoost**: `.json`, `.ubj`, `.pkl`
- **LightGBM**: `.txt`, `.pkl`
- **PyTorch**: `.pth`, `.pt`
- **ONNX**: `.onnx`

## Métricas Específicas do Lotofácil

O sistema coleta métricas específicas para modelos de Lotofácil:

- `acertos_15`: Taxa de acerto de 15 números
- `acertos_14`: Taxa de acerto de 14 números
- `acertos_13`: Taxa de acerto de 13 números
- `acertos_12`: Taxa de acerto de 12 números
- `acertos_11`: Taxa de acerto de 11 números
- `taxa_acerto_media`: Taxa média de acertos
- `precisao_numeros_frequentes`: Precisão em números frequentes
- `cobertura_padroes`: Cobertura de padrões identificados

## Testes

### Executar Testes Automatizados

```bash
python teste_sistema_backup.py
```

### Testes Incluídos

- ✅ Inicialização do sistema
- ✅ Criação de backups
- ✅ Listagem de versões
- ✅ Restauração de backups
- ✅ Validação de integridade
- ✅ Sistema completo

## Monitoramento e Logs

### Logs do Sistema

Os logs são salvos em `./backups_modelos/logs/backup_system.log`:

```
2024-01-15 10:30:00 - SistemaBackup - INFO - Backup criado: modelo.h5 -> 1.2.3
2024-01-15 10:35:00 - SistemaBackup - INFO - Verificação de integridade concluída
2024-01-15 11:00:00 - SistemaBackup - WARNING - Espaço em disco baixo: 500MB
```

### Relatórios

Relatórios são gerados automaticamente em:
- `./backups_modelos/relatorios/relatorio_diario.html`
- `./backups_modelos/relatorios/relatorio_diario.json`
- `./backups_modelos/relatorios/relatorio_diario.pdf`

## Integração com Outros Sistemas

### Git Integration

```json
{
  "integracao": {
    "git": {
      "habilitada": true,
      "auto_commit": true,
      "branch_backups": "backups"
    }
  }
}
```

### Cloud Storage

```json
{
  "integracao": {
    "cloud_storage": {
      "habilitado": true,
      "provedor": "aws_s3",
      "bucket": "meu-bucket-backups"
    }
  }
}
```

### Webhooks

```json
{
  "integracao": {
    "webhook": {
      "habilitado": true,
      "url": "https://meu-sistema.com/webhook",
      "eventos": ["backup_criado", "backup_falhou"]
    }
  }
}
```

## Solução de Problemas

### Problemas Comuns

#### Erro: "Arquivo não encontrado"
```bash
# Verificar se o arquivo existe
ls -la modelo.h5

# Verificar permissões
chmod 644 modelo.h5
```

#### Erro: "Espaço em disco insuficiente"
```bash
# Verificar espaço disponível
df -h

# Executar limpeza automática
python inicializar_backup.py limpeza
```

#### Erro: "Falha na validação"
```bash
# Verificar integridade do arquivo
python -c "from backup_versionamento import *; verificar_integridade('modelo.h5')"
```

### Logs de Debug

Para ativar logs detalhados:

```json
{
  "desenvolvimento": {
    "modo_debug": true,
    "log_detalhado": true
  }
}
```

## Performance

### Otimizações

- **Threads**: Backup e compressão em paralelo
- **Cache**: Cache de metadados para acesso rápido
- **Compressão**: Algoritmos otimizados para diferentes tipos de arquivo
- **Incremental**: Backup apenas de mudanças

### Configurações de Performance

```json
{
  "performance": {
    "threads_backup": 4,
    "threads_compressao": 2,
    "buffer_size_mb": 128,
    "cache_size_mb": 256
  }
}
```

## Segurança

### Criptografia

```json
{
  "seguranca": {
    "criptografia": {
      "habilitada": true,
      "algoritmo": "AES-256",
      "chave_arquivo": "./backups/.encryption_key"
    }
  }
}
```

### Controle de Acesso

```json
{
  "seguranca": {
    "controle_acesso": {
      "habilitado": true,
      "usuarios_autorizados": ["admin", "ml_engineer"],
      "log_acessos": true
    }
  }
}
```

## Contribuição

### Desenvolvimento

1. Clone o repositório
2. Instale dependências de desenvolvimento:
   ```bash
   pip install -r requirements_backup.txt
   ```
3. Execute os testes:
   ```bash
   python teste_sistema_backup.py
   ```
4. Faça suas modificações
5. Execute os testes novamente
6. Submeta um pull request

### Padrões de Código

- **Formatação**: Black com linha de 88 caracteres
- **Imports**: isort
- **Linting**: flake8
- **Type hints**: mypy
- **Documentação**: Docstrings em inglês

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## Suporte

Para suporte e dúvidas:

- 📧 Email: suporte@lotofacil-predicao.com
- 📖 Documentação: [docs.lotofacil-predicao.com](https://docs.lotofacil-predicao.com)
- 🐛 Issues: [github.com/lotofacil/issues](https://github.com/lotofacil/issues)

## Changelog

### v1.0.0 (2024-01-15)
- ✨ Implementação inicial do sistema
- ✨ Versionamento semântico
- ✨ Backup automático
- ✨ Validação de integridade
- ✨ API de linha de comando
- ✨ Testes automatizados
- ✨ Documentação completa

---

**Sistema de Backup e Versionamento - Lotofácil v1.0.0**  
*Desenvolvido para garantir a integridade e rastreabilidade dos modelos de machine learning.*