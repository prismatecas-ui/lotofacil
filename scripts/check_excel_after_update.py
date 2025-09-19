#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar as abas do Excel após atualização
"""

import pandas as pd

def main():
    print("🔍 VERIFICANDO EXCEL APÓS ATUALIZAÇÃO")
    print("=" * 45)
    
    try:
        xl = pd.ExcelFile('base/base_dados.xlsx')
        print(f"📋 Abas disponíveis: {xl.sheet_names}")
        
        for sheet in xl.sheet_names:
            df = pd.read_excel('base/base_dados.xlsx', sheet_name=sheet)
            print(f"\n📊 Aba '{sheet}': {len(df)} linhas")
            
            if 'Concurso' in df.columns:
                print(f"   🎯 Último concurso: {df['Concurso'].max()}")
                print(f"   📅 Primeira data: {df['Data Sorteio'].min() if 'Data Sorteio' in df.columns else 'N/A'}")
                print(f"   📅 Última data: {df['Data Sorteio'].max() if 'Data Sorteio' in df.columns else 'N/A'}")
            
            # Mostra primeiras colunas
            print(f"   📝 Colunas: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()