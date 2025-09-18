#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
print("=== TESTE DE AMBIENTE ===")
print(f"Python: {sys.version}")
print(f"Diretório atual: {os.getcwd()}")

try:
    import numpy as np
    print("✅ NumPy importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar NumPy: {e}")

try:
    import pandas as pd
    print("✅ Pandas importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar Pandas: {e}")

try:
    import tensorflow as tf
    print("✅ TensorFlow importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar TensorFlow: {e}")

try:
    import sqlite3
    print("✅ SQLite3 importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar SQLite3: {e}")

# Teste de conexão com banco
try:
    conn = sqlite3.connect('dados/lotofacil.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tabelas = cursor.fetchall()
    print(f"✅ Banco conectado. Tabelas: {[t[0] for t in tabelas]}")
    conn.close()
except Exception as e:
    print(f"❌ Erro no banco: {e}")

print("=== FIM DO TESTE ===")