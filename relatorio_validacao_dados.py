import json
import pandas as pd
from datetime import datetime
from dados.dados import carregar_dados

print("="*60)
print("RELATÓRIO DE VALIDAÇÃO DOS DADOS - LOTOFÁCIL")
print("="*60)
print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print()

# 1. VERIFICAR CACHE JSON
print("1. ANÁLISE DO CACHE JSON:")
print("-" * 30)
try:
    with open('./base/cache_concursos.json', 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    concursos_numeros = sorted([int(k) for k in cache_data.keys()])
    print(f"✅ Total de concursos no cache: {len(cache_data)}")
    print(f"✅ Concurso mais antigo: {concursos_numeros[0]}")
    print(f"✅ Concurso mais recente: {concursos_numeros[-1]}")
    print(f"✅ Período: {cache_data[str(concursos_numeros[0])]['Data Sorteio']} até {cache_data[str(concursos_numeros[-1])]['Data Sorteio']}")
    
    # Verificar estrutura dos dados
    primeiro_concurso = cache_data[str(concursos_numeros[0])]
    print(f"✅ Campos disponíveis: {len(primeiro_concurso)} campos")
    print(f"   - Dezenas: B1 a B15")
    print(f"   - Ganhadores: Sena, Quina, Quadra, Terno, Duque")
    print(f"   - Valores: Rateios e prêmios")
    print(f"   - Outros: Data, Acumulação, etc.")
    
except Exception as e:
    print(f"❌ Erro ao ler cache: {e}")

print()

# 2. VERIFICAR CARREGAMENTO DOS DADOS
print("2. TESTE DE CARREGAMENTO DOS DADOS:")
print("-" * 40)
try:
    dados = carregar_dados(usar_cache=True)
    print(f"✅ Dados carregados com sucesso")
    print(f"✅ Total de registros: {len(dados)}")
    print(f"✅ Colunas disponíveis: {list(dados.columns)}")
    print(f"✅ Período dos dados: Concurso {dados['Concurso'].min()} ao {dados['Concurso'].max()}")
    
    # Verificar qualidade dos dados
    print(f"✅ Dados faltantes por coluna:")
    missing_data = dados.isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"   - {col}: {missing} valores faltantes")
    
    if missing_data.sum() == 0:
        print("   - Nenhum dado faltante encontrado! 🎉")
    
    # Verificar consistência das dezenas
    dezenas_cols = [f'B{i}' for i in range(1, 16)]
    if all(col in dados.columns for col in dezenas_cols):
        print(f"✅ Todas as 15 colunas de dezenas estão presentes")
        
        # Verificar se as dezenas estão no range correto (1-25)
        for col in dezenas_cols:
            min_val = dados[col].min()
            max_val = dados[col].max()
            if min_val < 1 or max_val > 25:
                print(f"⚠️  {col}: valores fora do range (min: {min_val}, max: {max_val})")
            else:
                print(f"   - {col}: OK (range 1-25)")
    
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")

print()

# 3. VERIFICAR ÚLTIMOS CONCURSOS
print("3. ANÁLISE DOS ÚLTIMOS CONCURSOS:")
print("-" * 35)
try:
    # Mostrar os 5 últimos concursos
    ultimos = dados.tail(5)[['Concurso', 'Data Sorteio', 'Ganhadores_Sena', 'Ganhou']]
    print("Últimos 5 concursos:")
    for _, row in ultimos.iterrows():
        status = "COM GANHADOR" if row['Ganhou'] == 1 else "SEM GANHADOR"
        print(f"   - Concurso {row['Concurso']} ({row['Data Sorteio']}): {row['Ganhadores_Sena']} ganhadores - {status}")
    
except Exception as e:
    print(f"❌ Erro ao analisar últimos concursos: {e}")

print()

# 4. ESTATÍSTICAS GERAIS
print("4. ESTATÍSTICAS GERAIS:")
print("-" * 25)
try:
    total_concursos = len(dados)
    concursos_com_ganhador = dados['Ganhou'].sum()
    concursos_sem_ganhador = total_concursos - concursos_com_ganhador
    
    print(f"✅ Total de concursos: {total_concursos}")
    print(f"✅ Concursos com ganhador: {concursos_com_ganhador} ({concursos_com_ganhador/total_concursos*100:.1f}%)")
    print(f"✅ Concursos sem ganhador: {concursos_sem_ganhador} ({concursos_sem_ganhador/total_concursos*100:.1f}%)")
    
    # Verificar distribuição das dezenas
    dezenas_cols = [f'B{i}' for i in range(1, 16)]
    if all(col in dados.columns for col in dezenas_cols):
        todas_dezenas = dados[dezenas_cols].values.flatten()
        dezenas_unicas = sorted(set(todas_dezenas))
        print(f"✅ Dezenas encontradas: {dezenas_unicas}")
        print(f"✅ Range das dezenas: {min(dezenas_unicas)} a {max(dezenas_unicas)}")
    
except Exception as e:
    print(f"❌ Erro ao calcular estatísticas: {e}")

print()
print("="*60)
print("RESUMO DA VALIDAÇÃO:")
print("="*60)

try:
    # Verificações finais
    validacoes = []
    
    # 1. Cache existe e tem dados
    if len(cache_data) > 0:
        validacoes.append("✅ Cache JSON carregado com sucesso")
    else:
        validacoes.append("❌ Cache JSON vazio ou inexistente")
    
    # 2. Dados carregados corretamente
    if len(dados) > 0:
        validacoes.append(f"✅ {len(dados)} concursos carregados")
    else:
        validacoes.append("❌ Nenhum dado carregado")
    
    # 3. Dados atualizados
    if dados['Concurso'].max() >= 3490:
        validacoes.append("✅ Dados atualizados (concurso 3490+)")
    else:
        validacoes.append(f"⚠️  Dados podem estar desatualizados (último: {dados['Concurso'].max()})")
    
    # 4. Qualidade dos dados
    if dados.isnull().sum().sum() == 0:
        validacoes.append("✅ Nenhum dado faltante")
    else:
        validacoes.append(f"⚠️  {dados.isnull().sum().sum()} valores faltantes encontrados")
    
    for validacao in validacoes:
        print(validacao)
    
    print()
    print("RECOMENDAÇÃO:")
    if all("✅" in v for v in validacoes):
        print("🎉 DADOS VALIDADOS! Pode prosseguir com os testes.")
    else:
        print("⚠️  ATENÇÃO: Alguns problemas foram encontrados. Revisar antes de prosseguir.")
    
except Exception as e:
    print(f"❌ Erro na validação final: {e}")

print("="*60)