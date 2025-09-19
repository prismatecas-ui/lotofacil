import json
import pandas as pd
from datetime import datetime
import os

def validar_cache_concursos():
    """
    Valida o cache de concursos JSON e retorna estatísticas completas
    """
    print("🔍 Iniciando validação do cache de concursos...")
    
    try:
        # Carrega o cache JSON
        with open('./base/cache_concursos.json', 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        print(f"✅ Cache carregado com sucesso!")
        print(f"📊 Total de concursos no cache: {len(cache_data)}")
        
        # Converte para DataFrame para análise
        dados_list = []
        for concurso_num, dados_concurso in cache_data.items():
            dados_list.append(dados_concurso)
        
        df = pd.DataFrame(dados_list)
        df = df.sort_values('Concurso').reset_index(drop=True)
        
        # Estatísticas básicas
        print(f"\n📈 ESTATÍSTICAS DOS DADOS:")
        print(f"   • Concurso mais antigo: {df['Concurso'].min()}")
        print(f"   • Concurso mais recente: {df['Concurso'].max()}")
        print(f"   • Total de registros: {len(df)}")
        print(f"   • Data mais antiga: {df['Data Sorteio'].min()}")
        print(f"   • Data mais recente: {df['Data Sorteio'].max()}")
        
        # Verifica integridade dos dados
        print(f"\n🔍 VERIFICAÇÃO DE INTEGRIDADE:")
        
        # Verifica colunas essenciais
        colunas_essenciais = ['Concurso', 'Data Sorteio'] + [f'B{i}' for i in range(1, 16)] + ['Ganhadores_Sena']
        colunas_faltantes = [col for col in colunas_essenciais if col not in df.columns]
        
        if colunas_faltantes:
            print(f"   ❌ Colunas faltantes: {colunas_faltantes}")
        else:
            print(f"   ✅ Todas as colunas essenciais presentes")
        
        # Verifica valores nulos
        valores_nulos = df[colunas_essenciais].isnull().sum().sum()
        print(f"   • Valores nulos em colunas essenciais: {valores_nulos}")
        
        # Verifica range das bolas (1-25)
        bolas_cols = [f'B{i}' for i in range(1, 16)]
        bolas_invalidas = 0
        for col in bolas_cols:
            invalidas = ((df[col] < 1) | (df[col] > 25)).sum()
            bolas_invalidas += invalidas
        
        print(f"   • Bolas fora do range (1-25): {bolas_invalidas}")
        
        # Adiciona coluna 'Ganhou' para compatibilidade
        df['Ganhou'] = (df['Ganhadores_Sena'] > 0).astype(int)
        
        # Estatísticas de ganhadores
        concursos_com_ganhadores = (df['Ganhadores_Sena'] > 0).sum()
        concursos_sem_ganhadores = (df['Ganhadores_Sena'] == 0).sum()
        
        print(f"\n🎯 ESTATÍSTICAS DE GANHADORES:")
        print(f"   • Concursos com ganhadores: {concursos_com_ganhadores} ({concursos_com_ganhadores/len(df)*100:.1f}%)")
        print(f"   • Concursos sem ganhadores: {concursos_sem_ganhadores} ({concursos_sem_ganhadores/len(df)*100:.1f}%)")
        
        # Verifica distribuição das bolas
        print(f"\n🎲 ANÁLISE DE DISTRIBUIÇÃO DAS BOLAS:")
        todas_bolas = []
        for _, row in df.iterrows():
            bolas_sorteio = [row[f'B{i}'] for i in range(1, 16)]
            todas_bolas.extend(bolas_sorteio)
        
        freq_bolas = pd.Series(todas_bolas).value_counts().sort_index()
        print(f"   • Bola mais sorteada: {freq_bolas.idxmax()} ({freq_bolas.max()} vezes)")
        print(f"   • Bola menos sorteada: {freq_bolas.idxmin()} ({freq_bolas.min()} vezes)")
        print(f"   • Média de sorteios por bola: {freq_bolas.mean():.1f}")
        
        # Salva estatísticas
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        relatorio = {
            "timestamp": timestamp,
            "total_concursos": len(df),
            "concurso_min": int(df['Concurso'].min()),
            "concurso_max": int(df['Concurso'].max()),
            "data_min": df['Data Sorteio'].min(),
            "data_max": df['Data Sorteio'].max(),
            "integridade": {
                "colunas_faltantes": colunas_faltantes,
                "valores_nulos": int(valores_nulos),
                "bolas_invalidas": int(bolas_invalidas)
            },
            "ganhadores": {
                "com_ganhadores": int(concursos_com_ganhadores),
                "sem_ganhadores": int(concursos_sem_ganhadores),
                "percentual_com_ganhadores": round(concursos_com_ganhadores/len(df)*100, 2)
            },
            "distribuicao_bolas": {
                "mais_sorteada": int(freq_bolas.idxmax()),
                "freq_mais_sorteada": int(freq_bolas.max()),
                "menos_sorteada": int(freq_bolas.idxmin()),
                "freq_menos_sorteada": int(freq_bolas.min()),
                "media_sorteios": round(freq_bolas.mean(), 2)
            }
        }
        
        # Salva relatório
        os.makedirs('./experimentos/resultados', exist_ok=True)
        with open(f'./experimentos/resultados/validacao_cache_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Relatório salvo em: experimentos/resultados/validacao_cache_{timestamp}.json")
        
        # Retorna DataFrame para uso posterior
        return df, relatorio
        
    except FileNotFoundError:
        print("❌ Arquivo cache_concursos.json não encontrado!")
        return None, None
    except Exception as e:
        print(f"❌ Erro ao validar cache: {str(e)}")
        return None, None

if __name__ == "__main__":
    df, relatorio = validar_cache_concursos()
    
    if df is not None:
        print(f"\n✅ Validação concluída com sucesso!")
        print(f"📊 Dados prontos para uso: {len(df)} concursos validados")
    else:
        print(f"\n❌ Falha na validação dos dados")