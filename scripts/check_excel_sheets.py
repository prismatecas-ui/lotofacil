import pandas as pd

# Verificar abas disponíveis no Excel
xl = pd.ExcelFile('base/base_dados.xlsx')
print('Abas disponíveis:')
for sheet in xl.sheet_names:
    print(f'- {sheet}')
    
# Verificar tamanho de cada aba
for sheet in xl.sheet_names:
    try:
        df = pd.read_excel('base/base_dados.xlsx', sheet_name=sheet)
        print(f'\nAba "{sheet}": {len(df)} linhas')
        if len(df) > 0:
            print(f'Primeiras colunas: {list(df.columns[:5])}')
    except Exception as e:
        print(f'\nErro ao ler aba "{sheet}": {e}')