import pandas as pd
import os

def check_excel_update():
    excel_path = 'base/base_dados.xlsx'
    
    if not os.path.exists(excel_path):
        print(f"❌ Arquivo {excel_path} não encontrado")
        return
    
    try:
        # Ler o arquivo Excel
        df = pd.read_excel(excel_path)
        
        print(f"📊 Informações do arquivo Excel:")
        print(f"   - Total de linhas: {len(df)}")
        print(f"   - Total de colunas: {len(df.columns)}")
        
        if 'Concurso' in df.columns:
            concursos = df['Concurso'].dropna()
            print(f"   - Concursos únicos: {len(concursos.unique())}")
            print(f"   - Menor concurso: {concursos.min()}")
            print(f"   - Maior concurso: {concursos.max()}")
        
        # Verificar se existe cache
        cache_path = 'base/cache_concursos.json'
        if os.path.exists(cache_path):
            import json
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            print(f"\n📁 Cache de concursos:")
            print(f"   - Total de concursos no cache: {len(cache_data)}")
            
            cache_concursos = [int(k) for k in cache_data.keys()]
            print(f"   - Menor concurso no cache: {min(cache_concursos)}")
            print(f"   - Maior concurso no cache: {max(cache_concursos)}")
        
        print("\n✅ Verificação concluída com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo Excel: {e}")

if __name__ == "__main__":
    check_excel_update()