#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3

def check_database():
    try:
        conn = sqlite3.connect('dados/lotofacil.db')
        cursor = conn.cursor()
        
        # Verificar estrutura da tabela concursos
        cursor.execute("PRAGMA table_info(concursos);")
        columns = cursor.fetchall()
        print('Colunas da tabela concursos:')
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Verificar uma amostra dos dados
        cursor.execute("SELECT * FROM concursos LIMIT 2;")
        samples = cursor.fetchall()
        print('\nAmostra de dados:')
        for i, sample in enumerate(samples):
            print(f"Registro {i+1}: {sample}")
        
        # Verificar se existe coluna numeros_sorteados
        column_names = [col[1] for col in columns]
        if 'numeros_sorteados' in column_names:
            cursor.execute("SELECT numeros_sorteados FROM concursos LIMIT 1;")
            numeros_sample = cursor.fetchone()
            print(f"\nAmostra de numeros_sorteados: {numeros_sample[0]}")
            print(f"Tipo: {type(numeros_sample[0])}")
        
        conn.close()
        
    except Exception as e:
        print(f"Erro ao verificar banco: {e}")

if __name__ == '__main__':
    check_database()