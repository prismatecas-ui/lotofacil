# Guia de Acesso - Sistema Lotofácil Modernizado

## 📋 Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Conexão com internet (para atualizações da API da Caixa)

## 🚀 Primeiros Passos

### 1. Ativação do Ambiente Virtual

```bash
# Windows
cd c:\Users\braulio.augusto\Documents\Git\lotofacil
python -m venv venv
venv\Scripts\activate

# Linux/Mac
cd /caminho/para/lotofacil
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 3. Configuração do Banco de Dados

```bash
# Executar migrações (se necessário)
python scripts/migrate_to_sqlite.py
```

## 🎯 Executando o Sistema Principal

### Arquivo Principal - jogar.py

```bash
# Executar o sistema principal
python jogar.py
```

O arquivo `jogar.py` oferece um menu interativo com as seguintes opções:
- Análise de padrões
- Geração de jogos
- Consulta de resultados
- Estatísticas avançadas

## 🔮 API de Predições

### Iniciando o Servidor da API

```bash
# Executar servidor Flask
python api/prediction_api.py
```

### Endpoints Disponíveis

- **URL Base**: `http://localhost:5000`

#### Predições
```bash
# Obter predição para próximo concurso
GET http://localhost:5000/api/predict

# Obter predição com parâmetros específicos
POST http://localhost:5000/api/predict
Content-Type: application/json
{
  "model_type": "tensorflow",
  "num_predictions": 5
}
```

#### Resultados
```bash
# Obter último resultado
GET http://localhost:5000/api/results/latest

# Obter resultado específico
GET http://localhost:5000/api/results/{numero_concurso}
```

#### Estatísticas
```bash
# Estatísticas gerais
GET http://localhost:5000/api/stats

# Números mais sorteados
GET http://localhost:5000/api/stats/hot-numbers

# Números menos sorteados
GET http://localhost:5000/api/stats/cold-numbers
```

## 📊 Dashboard em Tempo Real

### Iniciando o Dashboard

```bash
# Executar dashboard web
python dashboard/app.py
```

- **URL de Acesso**: `http://localhost:8080`

### Funcionalidades do Dashboard

- **Estatísticas em Tempo Real**: Números mais/menos sorteados
- **Gráficos Interativos**: Frequência de números, tendências
- **Predições Visuais**: Resultados dos modelos de ML
- **Histórico de Concursos**: Navegação pelos resultados
- **Análise de Padrões**: Visualização de sequências e combinações

## 🔍 Scripts de Análise

### Análises Disponíveis

```bash
# Análise de frequência de números
python analises/analise_frequencia.py

# Análise de padrões sequenciais
python analises/analise_padroes.py

# Análise de combinações vencedoras
python analises/analise_combinacoes.py

# Análise estatística completa
python analises/analise_completa.py
```

### Relatórios Gerados

Os scripts geram relatórios em:
- `analises/relatorios/` - Relatórios em texto
- `analises/graficos/` - Gráficos e visualizações

## 🎲 Funcionalidades Avançadas

### Análise de Fechamentos

```bash
# Executar análise de fechamentos
python scripts/analise_fechamentos.py

# Gerar fechamentos otimizados
python scripts/gerar_fechamentos.py --numeros 15 --garantia 11
```

### Sistema de Desdobramentos

```bash
# Calcular desdobramentos
python scripts/desdobramentos.py --tipo completo --numeros 20

# Desdobramento com filtros
python scripts/desdobramentos.py --tipo filtrado --pares 7 --impares 8
```

### Gerador de Jogos Inteligente

```bash
# Gerar jogos baseados em IA
python scripts/gerador_inteligente.py --quantidade 10 --modelo tensorflow

# Gerar com parâmetros específicos
python scripts/gerador_inteligente.py --pares 8 --consecutivos 2 --primos 3
```

## 🔧 Comandos Essenciais

### Atualização de Dados

```bash
# Atualizar base de dados com últimos concursos
python scripts/update_data.py

# Sincronizar com API da Caixa
python scripts/sync_caixa_api.py

# Backup da base de dados
python scripts/backup_database.py
```

### Treinamento de Modelos

```bash
# Treinar modelo TensorFlow
python models/train_tensorflow_model.py

# Treinar modelo de ensemble
python models/train_ensemble_model.py

# Validar modelos existentes
python models/validate_models.py
```

### Manutenção do Sistema

```bash
# Verificar integridade dos dados
python scripts/check_data_integrity.py

# Limpar cache e arquivos temporários
python scripts/cleanup.py

# Executar testes do sistema
python -m pytest tests/
```

## 🌐 URLs de Acesso Rápido

| Serviço | URL | Descrição |
|---------|-----|----------|
| Dashboard Principal | `http://localhost:8080` | Interface web completa |
| API de Predições | `http://localhost:5000` | Endpoints REST |
| Documentação da API | `http://localhost:5000/docs` | Swagger UI |
| Monitoramento | `http://localhost:8080/monitor` | Status do sistema |
| Relatórios | `http://localhost:8080/reports` | Relatórios gerados |

## 📱 Uso Mobile/Responsivo

O dashboard é totalmente responsivo e pode ser acessado via:
- Navegadores desktop
- Tablets
- Smartphones

## 🔐 Configurações de Segurança

### Variáveis de Ambiente (.env)

```env
# Configurações da API
API_KEY=sua_chave_api_caixa
DATABASE_URL=sqlite:///base/lotofacil.db
SECRET_KEY=sua_chave_secreta

# Configurações do Dashboard
DASHBOARD_PORT=8080
API_PORT=5000
DEBUG=False
```

## 🆘 Solução de Problemas

### Problemas Comuns

1. **Erro de dependências**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Banco de dados corrompido**:
   ```bash
   python scripts/repair_database.py
   ```

3. **Cache desatualizado**:
   ```bash
   python scripts/clear_cache.py
   ```

4. **Modelos não encontrados**:
   ```bash
   python models/download_pretrained_models.py
   ```

### Logs do Sistema

- **Logs gerais**: `logs/system.log`
- **Logs da API**: `logs/api.log`
- **Logs de predições**: `logs/predictions.log`
- **Logs de erros**: `logs/errors.log`

## 📞 Suporte

Para suporte técnico:
1. Verifique os logs em `logs/`
2. Execute `python scripts/system_diagnostics.py`
3. Consulte a documentação em `docs/`

## 🎯 Fluxo de Uso Recomendado

1. **Primeira execução**:
   - Ativar ambiente virtual
   - Instalar dependências
   - Executar `python jogar.py`

2. **Uso diário**:
   - Atualizar dados: `python scripts/update_data.py`
   - Iniciar dashboard: `python dashboard/app.py`
   - Gerar predições: Acessar `http://localhost:8080`

3. **Análises avançadas**:
   - Executar scripts em `analises/`
   - Usar API para integrações
   - Consultar relatórios gerados

---

**Sistema Lotofácil Modernizado** - Versão 2.0

*Desenvolvido com Python, TensorFlow, Flask e tecnologias modernas de Machine Learning*