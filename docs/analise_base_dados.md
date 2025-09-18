# Análise da Estrutura do Arquivo base_dados.xlsx

## Informações Gerais
- **Arquivo**: base/base_dados.xlsx
- **Dimensões**: 1.994 linhas x 58 colunas
- **Formato**: Excel com múltiplas seções de dados

## Estrutura Identificada

### Seção 1: Dados dos Sorteios (Linhas 3-1993)
**Colunas principais:**
- **Coluna 0**: Vazia (índice)
- **Coluna 1**: Número do Concurso (1, 2, 3, ...)
- **Coluna 2**: Data do Sorteio (formato datetime)
- **Colunas 3-17**: Números sorteados (B1 a B15)
  - B1 a B15 representam as 15 dezenas sorteadas em cada concurso
  - Valores de 1 a 25 (números da Lotofácil)

### Seção 2: Análise de Ciclos (Colunas 19-45)
- **Coluna 19**: Marcador 'V' (possivelmente "Válido")
- **Colunas 20-44**: Números de 1 a 25 (análise de frequência por dezena)
- **Colunas 45-46**: Dados de ciclos (0-9)

### Seção 3: Análise de Faltantes (Colunas 47-57)
- **Coluna 47**: Indicador "Falta"
- **Coluna 48**: "Faltantes para completar o ciclo"
- **Colunas 49-57**: Dados de análise de padrões e faltantes

## Tipos de Dados Identificados

### Dados Principais dos Sorteios
```
Concurso | Data       | B1 | B2 | B3 | ... | B15
---------|------------|----|----|----|----|----
1        | 2003-09-29 | 18 | 20 | 25 | ... | 3
2        | 2003-10-06 | 23 | 15 | 5  | ... | 7
3        | 2003-10-13 | 20 | 23 | 12 | ... | 24
```

### Relacionamentos Identificados
1. **Concurso → Data**: Relação 1:1 (cada concurso tem uma data única)
2. **Concurso → Números**: Relação 1:15 (cada concurso tem 15 números)
3. **Análise de Frequência**: Colunas 20-44 contêm análise estatística das dezenas 1-25
4. **Análise de Ciclos**: Sistema de acompanhamento de padrões temporais

## Estrutura Proposta para SQLite

### Tabela: sorteios
```sql
CREATE TABLE sorteios (
    id INTEGER PRIMARY KEY,
    concurso INTEGER UNIQUE NOT NULL,
    data_sorteio DATE NOT NULL,
    numeros TEXT NOT NULL, -- JSON array com os 15 números
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Tabela: numeros_sorteados
```sql
CREATE TABLE numeros_sorteados (
    id INTEGER PRIMARY KEY,
    concurso INTEGER NOT NULL,
    numero INTEGER NOT NULL,
    posicao INTEGER NOT NULL, -- 1 a 15 (B1 a B15)
    FOREIGN KEY (concurso) REFERENCES sorteios(concurso)
);
```

### Tabela: estatisticas_dezenas
```sql
CREATE TABLE estatisticas_dezenas (
    id INTEGER PRIMARY KEY,
    dezena INTEGER NOT NULL, -- 1 a 25
    frequencia INTEGER DEFAULT 0,
    ultima_aparicao INTEGER, -- último concurso
    ciclo_atual INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Observações Importantes

1. **Dados Históricos Completos**: O arquivo contém dados desde o primeiro concurso (29/09/2003)
2. **Estrutura Complexa**: Múltiplas seções de análise além dos dados básicos
3. **Análise Estatística Integrada**: Sistema de ciclos e faltantes já implementado
4. **Formato Misto**: Combina dados transacionais com análises estatísticas

## Recomendações para Migração

1. **Separar Dados Transacionais**: Migrar sorteios e números para tabelas normalizadas
2. **Recalcular Estatísticas**: Gerar estatísticas a partir dos dados base
3. **Preservar Histórico**: Manter todos os 1.991 concursos identificados
4. **Validar Integridade**: Verificar se todos os sorteios têm exatamente 15 números únicos

## Status da Análise
✅ **CONCLUÍDO** - Estrutura do arquivo Excel completamente mapeada e documentada.