"""Script otimizado para atualizar dados do Excel com concursos da Lotofácil.

Este script utiliza requisições assíncronas e processamento em lotes
para acelerar significativamente a captura dos dados da API da Caixa.

Autor: Sistema de Upgrade Lotofácil
Data: 2025
"""

import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime
import time
import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading


class LotofacilUpdaterOptimized:
    """Classe otimizada para atualizar dados da Lotofácil com alta performance."""
    
    def __init__(self):
        self.base_url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
        self.excel_path = "base/base_dados.xlsx"
        self.backup_path = "base/backup_base_dados.xlsx"
        self.cache_file = "base/cache_concursos.json"
        self.max_concurrent = 10  # Máximo de requisições simultâneas
        self.batch_size = 50      # Tamanho do lote para processamento
        self.cache = {}
        self.lock = threading.Lock()
        
        # Headers para requisições
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
    
    def carregar_cache(self) -> None:
        """Carrega cache de concursos já processados."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"📦 Cache carregado: {len(self.cache)} concursos")
            else:
                self.cache = {}
                print("📦 Cache vazio - será criado")
        except Exception as e:
            print(f"⚠️  Erro ao carregar cache: {str(e)}")
            self.cache = {}
    
    def salvar_cache(self) -> None:
        """Salva cache de concursos processados."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            print(f"💾 Cache salvo: {len(self.cache)} concursos")
        except Exception as e:
            print(f"⚠️  Erro ao salvar cache: {str(e)}")
    
    def fazer_backup(self) -> bool:
        """Cria backup do arquivo Excel atual."""
        try:
            if os.path.exists(self.excel_path):
                import shutil
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"base/backup_base_dados_{timestamp}.xlsx"
                shutil.copy2(self.excel_path, backup_name)
                print(f"✅ Backup criado: {backup_name}")
                return True
            else:
                print("⚠️  Arquivo Excel não encontrado para backup")
                return False
        except Exception as e:
            print(f"❌ Erro ao criar backup: {str(e)}")
            return False
    
    async def obter_ultimo_concurso_api(self) -> Optional[int]:
        """Obtém o número do último concurso disponível na API."""
        try:
            print("🔍 Consultando último concurso disponível...")
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session:
                async with session.get(f"{self.base_url}/") as response:
                    if response.status == 200:
                        data = await response.json()
                        ultimo_concurso = data.get('numero', 0)
                        print(f"✅ Último concurso na API: {ultimo_concurso}")
                        return ultimo_concurso
                    else:
                        print(f"❌ Erro na API: Status {response.status}")
                        return None
                        
        except Exception as e:
            print(f"❌ Erro ao consultar API: {str(e)}")
            return None
    
    def obter_ultimo_concurso_excel(self) -> int:
        """Obtém o número do último concurso no arquivo Excel."""
        try:
            if not os.path.exists(self.excel_path):
                print("⚠️  Arquivo Excel não encontrado")
                return 0
            
            df = pd.read_excel(self.excel_path, header=None)
            
            # Procura pela coluna "Concurso"
            concurso_col = None
            for col in df.columns:
                if df.iloc[0, col] == 'Concurso' or str(df.iloc[0, col]).lower() == 'concurso':
                    concurso_col = col
                    break
            
            if concurso_col is not None:
                concursos = df.iloc[1:, concurso_col].dropna()
                concursos_numericos = pd.to_numeric(concursos, errors='coerce').dropna()
                
                if len(concursos_numericos) > 0:
                    ultimo_concurso = int(concursos_numericos.max())
                    print(f"✅ Último concurso no Excel: {ultimo_concurso}")
                    return ultimo_concurso
            
            print("⚠️  Não foi possível identificar concursos no Excel")
            return 0
            
        except Exception as e:
            print(f"❌ Erro ao ler Excel: {str(e)}")
            return 0
    
    async def obter_dados_concurso_async(self, session: aiohttp.ClientSession, numero_concurso: int) -> Optional[Dict]:
        """Obtém dados de um concurso específico da API de forma assíncrona."""
        try:
            # Verifica cache primeiro
            cache_key = str(numero_concurso)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            url = f"{self.base_url}/{numero_concurso}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    dados_processados = self.processar_dados_concurso(data)
                    
                    # Salva no cache
                    if dados_processados:
                        with self.lock:
                            self.cache[cache_key] = dados_processados
                    
                    return dados_processados
                else:
                    print(f"❌ Erro ao obter concurso {numero_concurso}: Status {response.status}")
                    return None
                    
        except Exception as e:
            print(f"❌ Erro ao obter concurso {numero_concurso}: {str(e)}")
            return None
    
    def processar_dados_concurso(self, data: Dict) -> Dict:
        """Processa os dados brutos da API para o formato do Excel."""
        try:
            # Extrai informações básicas
            concurso = data.get('numero', 0)
            data_sorteio = data.get('dataApuracao', '')
            
            # Converte data
            if data_sorteio:
                try:
                    data_obj = datetime.strptime(data_sorteio, '%d/%m/%Y')
                    data_formatada = data_obj.strftime('%d/%m/%Y')
                except:
                    data_formatada = data_sorteio
            else:
                data_formatada = ''
            
            # Extrai números sorteados
            numeros = data.get('listaDezenas', [])
            numeros_ordenados = sorted([int(n) for n in numeros])
            
            # Cria dicionário com dados do concurso
            dados_concurso = {
                'Concurso': concurso,
                'Data Sorteio': data_formatada,
            }
            
            # Adiciona os 15 números (B1 a B15)
            for i, numero in enumerate(numeros_ordenados, 1):
                dados_concurso[f'B{i}'] = numero
            
            # Preenche números faltantes com 0
            for i in range(len(numeros_ordenados) + 1, 16):
                dados_concurso[f'B{i}'] = 0
            
            # Adiciona informações de premiação
            premiacoes = data.get('listaRateioPremio', [])
            
            # Inicializa valores
            dados_concurso.update({
                'Ganhadores_Sena': 0,
                'Cidade': '',
                'UF': '',
                'Rateio_Sena': 0.0,
                'Ganhadores_Quina': 0,
                'Rateio_Quina': 0.0,
                'Ganhadores_Quadra': 0,
                'Rateio_Quadra': 0.0,
                'Ganhadores_Terno': 0,
                'Rateio_Terno': 0.0,
                'Ganhadores_Duque': 0,
                'Rateio_Duque': 0.0,
                'Acumulou': data.get('acumulou', False),
                'Valor_Acumulado': data.get('valorAcumuladoProximoConcurso', 0.0),
                'Estimativa_Premio': data.get('valorEstimadoProximoConcurso', 0.0),
                'Valor_Arrecadado': data.get('valorArrecadado', 0.0)
            })
            
            # Processa premiações
            for premiacao in premiacoes:
                acertos = premiacao.get('numeroDeGanhadores', 0)
                valor = premiacao.get('valorPremio', 0.0)
                faixa = premiacao.get('faixa', 0)
                
                if faixa == 1:  # 15 acertos
                    dados_concurso['Ganhadores_Sena'] = acertos
                    dados_concurso['Rateio_Sena'] = valor
                elif faixa == 2:  # 14 acertos
                    dados_concurso['Ganhadores_Quina'] = acertos
                    dados_concurso['Rateio_Quina'] = valor
                elif faixa == 3:  # 13 acertos
                    dados_concurso['Ganhadores_Quadra'] = acertos
                    dados_concurso['Rateio_Quadra'] = valor
                elif faixa == 4:  # 12 acertos
                    dados_concurso['Ganhadores_Terno'] = acertos
                    dados_concurso['Rateio_Terno'] = valor
                elif faixa == 5:  # 11 acertos
                    dados_concurso['Ganhadores_Duque'] = acertos
                    dados_concurso['Rateio_Duque'] = valor
            
            return dados_concurso
            
        except Exception as e:
            print(f"❌ Erro ao processar dados do concurso: {str(e)}")
            return None
    
    async def obter_lote_concursos(self, concursos: List[int]) -> List[Dict]:
        """Obtém dados de um lote de concursos de forma assíncrona."""
        dados_obtidos = []
        
        # Configura timeout e limites de conexão
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
            keepalive_timeout=30
        )
        
        async with aiohttp.ClientSession(
            timeout=timeout, 
            headers=self.headers,
            connector=connector
        ) as session:
            # Cria semáforo para limitar requisições simultâneas
            semaforo = asyncio.Semaphore(self.max_concurrent)
            
            async def obter_com_semaforo(concurso):
                async with semaforo:
                    return await self.obter_dados_concurso_async(session, concurso)
            
            # Executa todas as requisições em paralelo
            print(f"🚀 Processando lote de {len(concursos)} concursos...")
            inicio = time.time()
            
            tasks = [obter_com_semaforo(concurso) for concurso in concursos]
            resultados = await asyncio.gather(*tasks, return_exceptions=True)
            
            fim = time.time()
            tempo_total = fim - inicio
            
            # Processa resultados
            for i, resultado in enumerate(resultados):
                if isinstance(resultado, Exception):
                    print(f"❌ Erro no concurso {concursos[i]}: {resultado}")
                elif resultado:
                    dados_obtidos.append(resultado)
                    print(f"✅ Concurso {concursos[i]} processado")
            
            print(f"⚡ Lote processado em {tempo_total:.2f}s ({len(dados_obtidos)}/{len(concursos)} sucessos)")
            
        return dados_obtidos
    
    async def obter_novos_concursos_async(self, concurso_inicial: int, concurso_final: int) -> List[Dict]:
        """Obtém dados de múltiplos concursos usando processamento assíncrono em lotes."""
        todos_concursos = list(range(concurso_inicial, concurso_final + 1))
        total_concursos = len(todos_concursos)
        todos_dados = []
        
        print(f"🔄 Obtendo dados de {total_concursos} concurso(s) em lotes de {self.batch_size}...")
        
        # Processa em lotes
        for i in range(0, total_concursos, self.batch_size):
            lote = todos_concursos[i:i + self.batch_size]
            lote_num = (i // self.batch_size) + 1
            total_lotes = (total_concursos + self.batch_size - 1) // self.batch_size
            
            print(f"\n📦 Processando lote {lote_num}/{total_lotes} (concursos {lote[0]} a {lote[-1]})")
            
            dados_lote = await self.obter_lote_concursos(lote)
            todos_dados.extend(dados_lote)
            
            # Salva cache periodicamente
            if lote_num % 5 == 0:  # A cada 5 lotes
                self.salvar_cache()
            
            # Pequena pausa entre lotes para não sobrecarregar a API
            if i + self.batch_size < total_concursos:
                await asyncio.sleep(0.5)
        
        # Salva cache final
        self.salvar_cache()
        
        print(f"\n✅ Total de concursos obtidos: {len(todos_dados)}/{total_concursos}")
        return todos_dados
    
    def atualizar_excel_otimizado(self, novos_dados: List[Dict]) -> bool:
        """Atualiza o arquivo Excel com os novos dados de forma otimizada."""
        try:
            if not novos_dados:
                print("⚠️  Nenhum dado novo para atualizar")
                return True
            
            print(f"💾 Processando {len(novos_dados)} registros para o Excel...")
            
            # Lê o Excel atual
            if os.path.exists(self.excel_path):
                df_atual = pd.read_excel(self.excel_path)
                print(f"📊 Excel atual: {len(df_atual)} registros")
            else:
                colunas = list(novos_dados[0].keys())
                df_atual = pd.DataFrame(columns=colunas)
                print("📊 Criando novo arquivo Excel")
            
            # Converte novos dados para DataFrame
            df_novos = pd.DataFrame(novos_dados)
            
            # Remove duplicatas por concurso
            if 'Concurso' in df_atual.columns and len(df_atual) > 0:
                concursos_existentes = set(df_atual['Concurso'].tolist())
                df_novos = df_novos[~df_novos['Concurso'].isin(concursos_existentes)]
            
            if len(df_novos) > 0:
                # Combina os DataFrames
                df_final = pd.concat([df_atual, df_novos], ignore_index=True)
                
                # Ordena por concurso
                if 'Concurso' in df_final.columns:
                    df_final = df_final.sort_values('Concurso').reset_index(drop=True)
                
                # Salva o arquivo atualizado com otimizações
                with pd.ExcelWriter(self.excel_path, engine='openpyxl', options={'remove_timezone': True}) as writer:
                    df_final.to_excel(writer, index=False, sheet_name='Dados')
                
                print(f"✅ Excel atualizado com {len(df_novos)} novos registros")
                print(f"📊 Total de registros: {len(df_final)}")
                return True
            else:
                print("ℹ️  Todos os concursos já estão atualizados")
                return True
                
        except Exception as e:
            print(f"❌ Erro ao atualizar Excel: {str(e)}")
            return False
    
    async def executar_atualizacao_async(self) -> bool:
        """Executa o processo completo de atualização de forma assíncrona."""
        print("🚀 INICIANDO ATUALIZAÇÃO OTIMIZADA DOS DADOS DA LOTOFÁCIL")
        print("=" * 70)
        
        # Carrega cache
        self.carregar_cache()
        
        # Passo 1: Fazer backup
        print("\n📋 PASSO 1: Criando backup...")
        if not self.fazer_backup():
            print("⚠️  Continuando sem backup...")
        
        # Passo 2: Verificar último concurso na API
        print("\n🌐 PASSO 2: Consultando API da Caixa...")
        ultimo_concurso_api = await self.obter_ultimo_concurso_api()
        
        if not ultimo_concurso_api:
            print("❌ Não foi possível obter dados da API")
            return False
        
        # Passo 3: Verificar último concurso no Excel
        print("\n📊 PASSO 3: Analisando Excel atual...")
        ultimo_concurso_excel = self.obter_ultimo_concurso_excel()
        
        # Passo 4: Determinar concursos a atualizar
        print("\n🔍 PASSO 4: Determinando atualizações necessárias...")
        
        if ultimo_concurso_excel >= ultimo_concurso_api:
            print("✅ Excel já está atualizado!")
            print(f"   Excel: {ultimo_concurso_excel} | API: {ultimo_concurso_api}")
            return True
        
        concurso_inicial = ultimo_concurso_excel + 1
        concurso_final = ultimo_concurso_api
        
        print(f"📥 Concursos a atualizar: {concurso_inicial} até {concurso_final}")
        print(f"📊 Total de concursos: {concurso_final - concurso_inicial + 1}")
        
        # Passo 5: Obter novos dados (assíncrono)
        print("\n🔄 PASSO 5: Obtendo dados da API (modo otimizado)...")
        inicio_total = time.time()
        
        novos_dados = await self.obter_novos_concursos_async(concurso_inicial, concurso_final)
        
        fim_total = time.time()
        tempo_total = fim_total - inicio_total
        
        if not novos_dados:
            print("❌ Nenhum dado foi obtido da API")
            return False
        
        print(f"⚡ Dados obtidos em {tempo_total:.2f}s (média: {tempo_total/len(novos_dados):.2f}s por concurso)")
        
        # Passo 6: Atualizar Excel
        print("\n💾 PASSO 6: Atualizando arquivo Excel...")
        sucesso = self.atualizar_excel_otimizado(novos_dados)
        
        if sucesso:
            print("\n🎉 ATUALIZAÇÃO CONCLUÍDA COM SUCESSO!")
            print(f"✅ Arquivo atualizado: {self.excel_path}")
            print(f"📊 Último concurso: {ultimo_concurso_api}")
            print(f"⚡ Tempo total: {tempo_total:.2f}s")
            return True
        else:
            print("\n💥 FALHA NA ATUALIZAÇÃO")
            return False


def main():
    """Função principal."""
    updater = LotofacilUpdaterOptimized()
    
    try:
        # Executa a atualização assíncrona
        sucesso = asyncio.run(updater.executar_atualizacao_async())
        
        if sucesso:
            print("\n🚀 Dados atualizados e prontos para migração!")
        else:
            print("\n🔧 Verifique os erros e tente novamente.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Atualização interrompida pelo usuário")
    except Exception as e:
        print(f"\n💥 Erro inesperado: {str(e)}")


if __name__ == "__main__":
    main()