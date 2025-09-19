#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar se o Excel foi atualizado
"""

import pandas as pd

def main():
    print("🔍 VERIFICANDO EXCEL ATUALIZADO")
    print("=" * 40)
    
    try:
        # Verificar a aba Importar_Ciclo
        df = pd.read_excel('base/base_dados.xlsx', sheet_name='Importar_Ciclo')
        print(f"📊 Linhas na aba 'Importar_Ciclo': {len(df)}")
        print(f"🎯 Último concurso: {df['Concurso'].max()}")
        print(f"📅 Última data: {df['Data Sorteio'].max()}")
        
        # Verificar se há outras abas
        xl_file = pd.ExcelFile('base/base_dados.xlsx')
        print(f"📋 Abas disponíveis: {xl_file.sheet_names}")
        
        # Verificar cada aba
        for sheet in xl_file.sheet_names:
            df_sheet = pd.read_excel('base/base_dados.xlsx', sheet_name=sheet)
            print(f"   - {sheet}: {len(df_sheet)} linhas")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()