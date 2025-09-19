#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste para verificar se a correção dos dados funcionou
"""

import sys
sys.path.append('.')

from dados.dados import carregar_dados
import pandas as pd
from datetime import datetime

def main():
    print("🔧 TESTANDO CORREÇÃO DOS DADOS")
    print("=" * 40)
    
    # Testa carregamento com cache (novo método)
    print("\n📊 Testando carregamento com CACHE (corrigido):")
    dados_cache = carregar_dados(usar_cache=True)
    
    print(f"   📈 Total de concursos: {len(dados_cache)}")
    print(f"   🎯 Primeiro concurso: {dados_cache['Concurso'].min()}")
    print(f"   🎯 Último concurso: {dados_cache['Concurso'].max()}")
    
    if 'Data Sorteio' in dados_cache.columns:
        # Converte datas para análise
        dados_cache['Data_Parsed'] = pd.to_datetime(dados_cache['Data Sorteio'], format='%d/%m/%Y', errors='coerce')
        print(f"   📅 Primeira data: {dados_cache['Data_Parsed'].min().strftime('%d/%m/%Y')}")
        print(f"   📅 Última data: {dados_cache['Data_Parsed'].max().strftime('%d/%m/%Y')}")
        
        # Verifica concursos de 2024
        concursos_2024 = dados_cache[dados_cache['Data_Parsed'].dt.year == 2024]
        print(f"   🎊 Concursos de 2024: {len(concursos_2024)}")
        
        if len(concursos_2024) > 0:
            print(f"   🎊 Primeiro de 2024: {concursos_2024['Concurso'].min()} ({concursos_2024['Data_Parsed'].min().strftime('%d/%m/%Y')})")
            print(f"   🎊 Último de 2024: {concursos_2024['Concurso'].max()} ({concursos_2024['Data_Parsed'].max().strftime('%d/%m/%Y')})")
    
    # Verifica se tem coluna 'Ganhou'
    if 'Ganhou' in dados_cache.columns:
        ganhadores = dados_cache['Ganhou'].sum()
        print(f"   🏆 Concursos com ganhadores: {ganhadores}")
    
    print(f"   📝 Colunas disponíveis: {list(dados_cache.columns[:10])}{'...' if len(dados_cache.columns) > 10 else ''}")
    
    # Compara com método antigo
    print("\n📊 Comparando com método ANTIGO (Excel):")
    dados_excel = carregar_dados(usar_cache=False)
    print(f"   📈 Total Excel: {len(dados_excel)} vs Cache: {len(dados_cache)}")
    print(f"   📈 Diferença: +{len(dados_cache) - len(dados_excel)} concursos")
    
    print("\n✅ TESTE CONCLUÍDO!")
    
    if len(dados_cache) > 3000:
        print("🎉 PROBLEMA RESOLVIDO! Dataset agora tem dados completos!")
    else:
        print("❌ Ainda há problemas no carregamento dos dados.")

if __name__ == "__main__":
    main()