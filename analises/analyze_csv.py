import pandas as pd

# Carregar o CSV com separador correto
df = pd.read_csv('base/resultados.csv', sep=';')

print('=== ESTRUTURA DO CSV RESULTADOS ===\n')
print(f'Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas')
print(f'Período: {df["Data Sorteio"].iloc[0]} até {df["Data Sorteio"].iloc[-1]}')
print(f'Concursos: {df["Concurso"].min()} até {df["Concurso"].max()}')

print('\n=== COLUNAS ===\n')
for i, col in enumerate(df.columns, 1):
    print(f'{i:2d}. {col} ({df[col].dtype})')

print('\n=== ANÁLISE DA COLUNA GANHOU ===\n')
print('Valores únicos:', sorted(df['Ganhou'].unique()))
print('\nDistribuição de valores:')
print(df['Ganhou'].value_counts().sort_index())

print('\n=== ANÁLISE DOS NÚMEROS SORTEADOS ===\n')
numeros_cols = [f'B{i}' for i in range(1, 16)]
print(f'Faixa de números: {df[numeros_cols].min().min()} a {df[numeros_cols].max().max()}')
print(f'Total de colunas de números: {len(numeros_cols)}')

print('\n=== PRIMEIRAS 3 LINHAS ===\n')
print(df.head(3).to_string())

print('\n=== ÚLTIMAS 3 LINHAS ===\n')
print(df.tail(3).to_string())

print('\n=== VERIFICAÇÃO DE INTEGRIDADE ===\n')
print('Valores nulos por coluna:')
print(df.isnull().sum())

print('\nVerificação de números válidos (1-25):')
for col in numeros_cols:
    invalid = df[(df[col] < 1) | (df[col] > 25)]
    if len(invalid) > 0:
        print(f'ERRO: {col} tem valores inválidos: {invalid[col].unique()}')
    else:
        print(f'OK: {col} - todos os valores entre 1-25')

print('\n=== ANÁLISE ESTATÍSTICA ===\n')
print('Estatísticas da coluna Ganhou:')
print(df['Ganhou'].describe())