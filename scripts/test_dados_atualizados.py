#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para testar se os dados foram atualizados corretamente
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dados.dados import carregar_dados
import pandas as pd

def main():
    print("🔍 TESTANDO DADOS ATUALIZADOS")
    print("=" * 50)
    
    try:
        # Carregar dados
        print("📊 Carregando dados...")
        df = carregar_dados()
        
        print(f"✅ Total de concursos carregados: {len(df)}")
        print(f"📅 Primeiro concurso: {df['Data Sorteio'].min()}")
        print(f"📅 Último concurso: {df['Data Sorteio'].max()}")
        print(f"🎯 Último número de concurso: {df['Concurso'].max()}")
        
        # Verificar se temos dados recentes (2024)
        df['Data Sorteio'] = pd.to_datetime(df['Data Sorteio'])
        dados_2024 = df[df['Data Sorteio'].dt.year == 2024]
        print(f"📈 Concursos de 2024: {len(dados_2024)}")
        
        if len(df) > 3000:
            print("🎉 SUCESSO! Dataset atualizado com dados completos!")
        else:
            print("⚠️  ATENÇÃO: Dataset ainda parece limitado")
            
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()