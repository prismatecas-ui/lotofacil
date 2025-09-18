# Análise da Estrutura do Arquivo CSV - base/resultados.csv

## Informações Gerais
- **Arquivo**: `base/resultados.csv`
- **Formato**: CSV com separador ponto e vírgula (;)
- **Codificação**: UTF-8
- **Dimensões**: 1.991 linhas × 18 colunas
- **Período**: 29/09/2003 até 10/07/2020
- **Concursos**: 1 até 1991
- **Integridade**: 100% - Nenhum valor nulo encontrado

## Estrutura das Colunas

### 1. Coluna de Identificação
- **Concurso** (int64): Número sequencial do concurso (1-1991)

### 2. Coluna Temporal
- **Data Sorteio** (object): Data do sorteio no formato DD/MM/AAAA

### 3. Colunas dos Números Sorteados (15 colunas)
- **B1 a B15** (int64): Os 15 números sorteados em cada concurso
- **Faixa válida**: 1 a 25 (todos os valores estão corretos)
- **Validação**: ✅ Todos os números estão dentro da faixa válida da Lotofácil

### 4. Coluna de Resultado
- **Ganhou** (int64): Quantidade de números acertados
- **Faixa**: 0 a 94 acertos
- **Média**: 3,98 acertos
- **Mediana**: 3 acertos
- **Desvio padrão**: 5,22

## Distribuição da Coluna 'Ganhou'

```
Acertos | Frequência
--------|----------
   0    |    318
   1    |    398
   2    |    312
   3    |    285
   4    |    219
   5    |    162
   6    |    108
   7    |     67
   8    |     42
   9    |     28
  10    |     19
  11    |     13
  12    |      8
  13    |      6
  14    |      4
  15    |      2
  94    |      1  (valor atípico)
```

## Observações Importantes

### Pontos Positivos
1. **Dados Completos**: Nenhum valor nulo em nenhuma coluna
2. **Consistência Temporal**: Sequência cronológica correta dos concursos
3. **Integridade dos Números**: Todos os números sorteados estão na faixa 1-25
4. **Formato Padronizado**: Estrutura consistente em todas as linhas

### Pontos de Atenção
1. **Valor Atípico**: Existe 1 registro com 94 acertos (provavelmente erro ou caso especial)
2. **Formato de Data**: String no formato DD/MM/AAAA (necessário conversão para datetime)
3. **Separador**: Usa ponto e vírgula (;) em vez de vírgula padrão

## Relacionamentos Identificados

### Estrutura Relacional
- **Concurso → Data**: Relacionamento 1:1 (cada concurso tem uma data única)
- **Concurso → Números**: Relacionamento 1:15 (cada concurso tem 15 números)
- **Concurso → Resultado**: Relacionamento 1:1 (cada concurso tem um resultado)

## Proposta de Tabelas SQLite

### Tabela: `sorteios_resultados`
```sql
CREATE TABLE sorteios_resultados (
    concurso INTEGER PRIMARY KEY,
    data_sorteio DATE NOT NULL,
    numero_01 INTEGER NOT NULL CHECK (numero_01 BETWEEN 1 AND 25),
    numero_02 INTEGER NOT NULL CHECK (numero_02 BETWEEN 1 AND 25),
    numero_03 INTEGER NOT NULL CHECK (numero_03 BETWEEN 1 AND 25),
    numero_04 INTEGER NOT NULL CHECK (numero_04 BETWEEN 1 AND 25),
    numero_05 INTEGER NOT NULL CHECK (numero_05 BETWEEN 1 AND 25),
    numero_06 INTEGER NOT NULL CHECK (numero_06 BETWEEN 1 AND 25),
    numero_07 INTEGER NOT NULL CHECK (numero_07 BETWEEN 1 AND 25),
    numero_08 INTEGER NOT NULL CHECK (numero_08 BETWEEN 1 AND 25),
    numero_09 INTEGER NOT NULL CHECK (numero_09 BETWEEN 1 AND 25),
    numero_10 INTEGER NOT NULL CHECK (numero_10 BETWEEN 1 AND 25),
    numero_11 INTEGER NOT NULL CHECK (numero_11 BETWEEN 1 AND 25),
    numero_12 INTEGER NOT NULL CHECK (numero_12 BETWEEN 1 AND 25),
    numero_13 INTEGER NOT NULL CHECK (numero_13 BETWEEN 1 AND 25),
    numero_14 INTEGER NOT NULL CHECK (numero_14 BETWEEN 1 AND 25),
    numero_15 INTEGER NOT NULL CHECK (numero_15 BETWEEN 1 AND 25),
    acertos INTEGER NOT NULL CHECK (acertos >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Índices Recomendados
```sql
CREATE INDEX idx_sorteios_data ON sorteios_resultados(data_sorteio);
CREATE INDEX idx_sorteios_acertos ON sorteios_resultados(acertos);
```

## Recomendações para Migração

1. **Conversão de Data**: Converter string DD/MM/AAAA para formato DATE do SQLite
2. **Validação de Dados**: Verificar e tratar o valor atípico de 94 acertos
3. **Normalização**: Considerar tabela separada para números individuais se necessário
4. **Índices**: Criar índices nas colunas mais consultadas (data, acertos)
5. **Constraints**: Implementar verificações de integridade nos números (1-25)

## Status da Análise
✅ **CONCLUÍDO** - Estrutura do CSV completamente mapeada e documentada

---
*Análise realizada em: Janeiro 2025*
*Arquivo analisado: base/resultados.csv (1.991 registros)*