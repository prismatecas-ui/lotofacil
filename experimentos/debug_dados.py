import pandas as pd
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dados.dados import carregar_dados

def debug_dados():
    """Debug dos dados carregados"""
    print("Carregando dados para debug...")
    
    try:
        # Carregar dados
        dados = carregar_dados('Importar_Ciclo')
        
        print(f"\nShape dos dados: {dados.shape}")
        print(f"\nColunas: {list(dados.columns)}")
        print(f"\nPrimeiras 5 linhas:")
        print(dados.head())
        
        print(f"\nÚltimas 5 linhas:")
        print(dados.tail())
        
        print(f"\nTipos de dados:")
        print(dados.dtypes)
        
        print(f"\nValores nulos por coluna:")
        print(dados.isnull().sum())
        
        # Verificar colunas numéricas
        numeric_cols = dados.select_dtypes(include=['number']).columns
        print(f"\nColunas numéricas: {list(numeric_cols)}")
        
        # Verificar se há colunas de números da lotofácil
        possible_number_cols = [col for col in dados.columns if any(x in col.lower() for x in ['bola', 'numero', 'dezena'])]
        print(f"\nPossíveis colunas de números: {possible_number_cols}")
        
        # Verificar estatísticas básicas
        if len(numeric_cols) > 0:
            print(f"\nEstatísticas das colunas numéricas:")
            print(dados[numeric_cols].describe())
        
        return dados
        
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

if __name__ == "__main__":
    dados = debug_dados()