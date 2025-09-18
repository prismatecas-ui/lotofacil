import pandas as pd
import os

# Mudar para o diretório do projeto
os.chdir('c:\\Users\\braulio.augusto\\Documents\\Git\\lotofacil')

# Carregar o arquivo Excel sem header para ver a estrutura real
df_raw = pd.read_excel('base/base_dados.xlsx', header=None)

print('=== ANÁLISE DETALHADA DO ARQUIVO base_dados.xlsx ===')
print(f'Dimensões: {df_raw.shape[0]} linhas x {df_raw.shape[1]} colunas')

print('\n=== PRIMEIRAS 10 LINHAS (SEM HEADER) ===')
print(df_raw.head(10))

print('\n=== ANÁLISE DAS PRIMEIRAS LINHAS PARA IDENTIFICAR ESTRUTURA ===')
for i in range(min(15, len(df_raw))):
    row_data = df_raw.iloc[i].dropna().tolist()
    if row_data:  # Se a linha não está vazia
        print(f'Linha {i}: {row_data[:10]}...' if len(row_data) > 10 else f'Linha {i}: {row_data}')

# Tentar identificar onde começam os dados dos sorteios
print('\n=== PROCURANDO INÍCIO DOS DADOS DE SORTEIOS ===')
for i in range(len(df_raw)):
    row = df_raw.iloc[i]
    # Procurar por números que parecem ser de concursos
    if pd.notna(row.iloc[1]) and str(row.iloc[1]).isdigit():
        if int(row.iloc[1]) == 1:  # Primeiro concurso
            print(f'Dados de sorteios começam na linha {i}')
            print(f'Exemplo da linha: {row.dropna().tolist()}')
            break

# Analisar uma seção específica que parece conter dados de sorteios
print('\n=== ANÁLISE DE UMA SEÇÃO DE DADOS ===')
start_row = 3  # Baseado na análise anterior
end_row = min(20, len(df_raw))
sample_data = df_raw.iloc[start_row:end_row]
print(sample_data)

print('\n=== IDENTIFICAÇÃO DE COLUNAS RELEVANTES ===')
# Analisar as colunas que parecem conter números dos sorteios
for col_idx in range(min(25, df_raw.shape[1])):
    col_data = df_raw.iloc[3:20, col_idx].dropna()
    if len(col_data) > 0:
        print(f'Coluna {col_idx}: {col_data.tolist()[:5]}...')