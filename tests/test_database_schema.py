import sqlite3
import os

def test_database_schema():
    """Testa a criação do esquema do banco de dados SQLite"""
    
    # Caminho do arquivo de teste
    test_db_path = 'database/test_schema.db'
    sql_script_path = 'database/create_tables.sql'
    
    try:
        # Remove banco de teste se existir
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print(f"✅ Arquivo de teste removido: {test_db_path}")
        
        # Lê o script SQL
        with open(sql_script_path, 'r', encoding='utf-8') as file:
            sql_script = file.read()
            print(f"✅ Script SQL carregado: {len(sql_script)} caracteres")
        
        # Conecta ao banco e executa o script
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # Executa o script completo
        cursor.executescript(sql_script)
        print("✅ Script SQL executado com sucesso")
        
        # Verifica as tabelas criadas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()
        
        print(f"\n=== TABELAS CRIADAS ({len(tables)}) ===")
        for i, (table_name,) in enumerate(tables, 1):
            print(f"{i:2d}. {table_name}")
            
            # Conta registros em cada tabela
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"    Registros: {count}")
        
        # Verifica os índices criados
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        indexes = cursor.fetchall()
        
        print(f"\n=== ÍNDICES CRIADOS ({len(indexes)}) ===")
        for i, (index_name,) in enumerate(indexes, 1):
            print(f"{i:2d}. {index_name}")
        
        # Verifica as views criadas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;")
        views = cursor.fetchall()
        
        print(f"\n=== VIEWS CRIADAS ({len(views)}) ===")
        for i, (view_name,) in enumerate(views, 1):
            print(f"{i:2d}. {view_name}")
        
        # Verifica os triggers criados
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name;")
        triggers = cursor.fetchall()
        
        print(f"\n=== TRIGGERS CRIADOS ({len(triggers)}) ===")
        for i, (trigger_name,) in enumerate(triggers, 1):
            print(f"{i:2d}. {trigger_name}")
        
        # Verifica configurações iniciais
        cursor.execute("SELECT chave, valor, descricao FROM configuracoes ORDER BY chave;")
        configs = cursor.fetchall()
        
        print(f"\n=== CONFIGURAÇÕES INICIAIS ({len(configs)}) ===")
        for chave, valor, descricao in configs:
            print(f"  {chave}: {valor} ({descricao})")
        
        # Verifica estatísticas dos números
        cursor.execute("SELECT COUNT(*) FROM estatisticas_numeros;")
        stats_count = cursor.fetchone()[0]
        print(f"\n=== ESTATÍSTICAS DOS NÚMEROS ===")
        print(f"  Números inicializados: {stats_count}/25")
        
        conn.close()
        print(f"\n✅ TESTE CONCLUÍDO COM SUCESSO!")
        print(f"   Banco de teste criado: {test_db_path}")
        print(f"   Tamanho do arquivo: {os.path.getsize(test_db_path)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO durante o teste: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== TESTE DO ESQUEMA DO BANCO DE DADOS ===")
    print("Testando criação das tabelas SQLite...\n")
    
    success = test_database_schema()
    
    if success:
        print("\n🎉 Esquema do banco validado com sucesso!")
    else:
        print("\n💥 Falha na validação do esquema!")