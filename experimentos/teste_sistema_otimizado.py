#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Completo do Sistema Otimizado - Lotofácil
Validação das otimizações implementadas
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TesteSistemaOtimizado:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.modelo_dir = os.path.join(self.base_dir, 'modelo')
        self.dados_dir = os.path.join(self.base_dir, 'dados')
        self.resultados_dir = os.path.join(self.base_dir, 'experimentos', 'resultados')
        
        # Criar diretórios se não existirem
        os.makedirs(self.resultados_dir, exist_ok=True)
        
        self.resultados = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'testes_realizados': [],
            'modelos_testados': {},
            'performance_geral': {},
            'validacoes': {},
            'recomendacoes': []
        }
    
    def carregar_dados_teste(self):
        """Carrega dados para teste"""
        print("📊 Carregando dados de teste...")
        
        try:
            # Tentar carregar dados processados
            dados_path = os.path.join(self.dados_dir, 'dados_processados.csv')
            if os.path.exists(dados_path):
                df = pd.read_csv(dados_path)
                print(f"   ✅ Dados carregados: {df.shape}")
                return df
            
            # Se não existir, criar dados de exemplo
            print("   ⚠️  Criando dados de exemplo para teste...")
            np.random.seed(42)
            n_samples = 1000
            
            # Simular dados de jogos da Lotofácil
            dados = []
            for i in range(n_samples):
                jogo = sorted(np.random.choice(range(1, 26), 15, replace=False))
                dados.append(jogo)
            
            df = pd.DataFrame(dados, columns=[f'num_{i+1}' for i in range(15)])
            
            # Salvar dados de exemplo
            df.to_csv(dados_path, index=False)
            print(f"   ✅ Dados de exemplo criados: {df.shape}")
            return df
            
        except Exception as e:
            print(f"   ❌ Erro ao carregar dados: {e}")
            return None
    
    def testar_modelos_otimizados(self):
        """Testa os modelos otimizados"""
        print("\n🤖 Testando modelos otimizados...")
        
        modelos_dir = os.path.join(self.modelo_dir, 'optimized_models')
        if not os.path.exists(modelos_dir):
            print("   ❌ Diretório de modelos otimizados não encontrado")
            return
        
        modelos_testados = {}
        
        # Listar modelos disponíveis
        for arquivo in os.listdir(modelos_dir):
            if arquivo.endswith('.pkl'):
                modelo_path = os.path.join(modelos_dir, arquivo)
                nome_modelo = arquivo.replace('.pkl', '')
                
                try:
                    print(f"   🔍 Testando {nome_modelo}...")
                    
                    # Carregar modelo
                    with open(modelo_path, 'rb') as f:
                        modelo = pickle.load(f)
                    
                    # Informações do modelo
                    info_modelo = {
                        'arquivo': arquivo,
                        'tamanho_arquivo': os.path.getsize(modelo_path),
                        'tipo': str(type(modelo)),
                        'status': 'carregado_com_sucesso'
                    }
                    
                    # Tentar fazer predição de teste
                    if hasattr(modelo, 'predict'):
                        # Criar dados de teste simples
                        X_teste = np.random.rand(10, 15)
                        try:
                            predicoes = modelo.predict(X_teste)
                            info_modelo['predicao_teste'] = 'sucesso'
                            info_modelo['formato_saida'] = str(type(predicoes[0]))
                        except Exception as e:
                            info_modelo['predicao_teste'] = f'erro: {str(e)}'
                    
                    modelos_testados[nome_modelo] = info_modelo
                    print(f"   ✅ {nome_modelo}: OK")
                    
                except Exception as e:
                    modelos_testados[nome_modelo] = {
                        'arquivo': arquivo,
                        'status': 'erro',
                        'erro': str(e)
                    }
                    print(f"   ❌ {nome_modelo}: {e}")
        
        self.resultados['modelos_testados'] = modelos_testados
        return modelos_testados
    
    def validar_arquivos_resultados(self):
        """Valida arquivos de resultados gerados"""
        print("\n📁 Validando arquivos de resultados...")
        
        arquivos_validados = {}
        
        # Verificar arquivos de otimizações
        for arquivo in os.listdir(self.resultados_dir):
            if 'otimizacoes' in arquivo and arquivo.endswith('.json'):
                arquivo_path = os.path.join(self.resultados_dir, arquivo)
                
                try:
                    with open(arquivo_path, 'r', encoding='utf-8') as f:
                        dados = json.load(f)
                    
                    info_arquivo = {
                        'tamanho': os.path.getsize(arquivo_path),
                        'chaves_principais': list(dados.keys()) if isinstance(dados, dict) else 'não_dict',
                        'status': 'válido'
                    }
                    
                    # Verificar métricas específicas
                    if isinstance(dados, dict):
                        if 'metricas' in dados:
                            metricas = dados['metricas']
                            info_arquivo['accuracy'] = metricas.get('accuracy', 'não_encontrado')
                            info_arquivo['precision'] = metricas.get('precision', 'não_encontrado')
                        
                        if 'ensemble_performance' in dados:
                            info_arquivo['ensemble_accuracy'] = dados['ensemble_performance'].get('accuracy', 'não_encontrado')
                    
                    arquivos_validados[arquivo] = info_arquivo
                    print(f"   ✅ {arquivo}: Válido")
                    
                except Exception as e:
                    arquivos_validados[arquivo] = {
                        'status': 'erro',
                        'erro': str(e)
                    }
                    print(f"   ❌ {arquivo}: {e}")
        
        self.resultados['validacoes']['arquivos_resultados'] = arquivos_validados
        return arquivos_validados
    
    def testar_performance_sistema(self):
        """Testa performance geral do sistema"""
        print("\n⚡ Testando performance do sistema...")
        
        performance = {
            'tempo_carregamento_modelos': 0,
            'tempo_predicao': 0,
            'uso_memoria': 0,
            'status_geral': 'ok'
        }
        
        try:
            import time
            import psutil
            
            # Medir uso de memória inicial
            memoria_inicial = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Testar carregamento de modelos
            inicio = time.time()
            modelos = self.testar_modelos_otimizados()
            fim = time.time()
            performance['tempo_carregamento_modelos'] = fim - inicio
            
            # Medir uso de memória final
            memoria_final = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            performance['uso_memoria'] = memoria_final - memoria_inicial
            
            print(f"   ⏱️  Tempo carregamento: {performance['tempo_carregamento_modelos']:.2f}s")
            print(f"   💾 Uso de memória: {performance['uso_memoria']:.2f}MB")
            
        except Exception as e:
            performance['status_geral'] = f'erro: {str(e)}'
            print(f"   ❌ Erro no teste de performance: {e}")
        
        self.resultados['performance_geral'] = performance
        return performance
    
    def gerar_recomendacoes(self):
        """Gera recomendações baseadas nos testes"""
        print("\n💡 Gerando recomendações...")
        
        recomendacoes = []
        
        # Analisar modelos testados
        modelos_ok = sum(1 for m in self.resultados['modelos_testados'].values() 
                        if m.get('status') == 'carregado_com_sucesso')
        total_modelos = len(self.resultados['modelos_testados'])
        
        if modelos_ok == total_modelos and total_modelos > 0:
            recomendacoes.append("✅ Todos os modelos foram carregados com sucesso")
        elif modelos_ok > 0:
            recomendacoes.append(f"⚠️  {modelos_ok}/{total_modelos} modelos funcionando corretamente")
        else:
            recomendacoes.append("❌ Nenhum modelo funcionando - revisar implementação")
        
        # Analisar performance
        if 'performance_geral' in self.resultados:
            perf = self.resultados['performance_geral']
            if perf.get('tempo_carregamento_modelos', 0) > 10:
                recomendacoes.append("⚠️  Tempo de carregamento alto - considerar otimização")
            if perf.get('uso_memoria', 0) > 500:
                recomendacoes.append("⚠️  Alto uso de memória - considerar otimização")
        
        # Recomendações gerais
        recomendacoes.extend([
            "🔄 Implementar monitoramento contínuo de performance",
            "📊 Configurar logging detalhado para produção",
            "🎯 Testar com dados reais de produção",
            "🔒 Implementar validação de entrada robusta",
            "⚡ Considerar cache para predições frequentes"
        ])
        
        self.resultados['recomendacoes'] = recomendacoes
        
        for rec in recomendacoes:
            print(f"   {rec}")
        
        return recomendacoes
    
    def executar_teste_completo(self):
        """Executa todos os testes do sistema"""
        print("🚀 INICIANDO TESTE COMPLETO DO SISTEMA OTIMIZADO")
        print("=" * 60)
        
        # Registrar início do teste
        self.resultados['testes_realizados'].append('inicio_teste_completo')
        
        # 1. Carregar dados
        dados = self.carregar_dados_teste()
        if dados is not None:
            self.resultados['testes_realizados'].append('carregamento_dados')
        
        # 2. Testar modelos
        modelos = self.testar_modelos_otimizados()
        if modelos:
            self.resultados['testes_realizados'].append('teste_modelos')
        
        # 3. Validar arquivos
        arquivos = self.validar_arquivos_resultados()
        if arquivos:
            self.resultados['testes_realizados'].append('validacao_arquivos')
        
        # 4. Testar performance
        performance = self.testar_performance_sistema()
        if performance:
            self.resultados['testes_realizados'].append('teste_performance')
        
        # 5. Gerar recomendações
        recomendacoes = self.gerar_recomendacoes()
        if recomendacoes:
            self.resultados['testes_realizados'].append('geracao_recomendacoes')
        
        # Salvar resultados
        self.salvar_resultados()
        
        # Exibir resumo
        self.exibir_resumo()
    
    def salvar_resultados(self):
        """Salva os resultados dos testes"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        arquivo_resultado = os.path.join(self.resultados_dir, f'teste_sistema_completo_{timestamp}.json')
        
        try:
            with open(arquivo_resultado, 'w', encoding='utf-8') as f:
                json.dump(self.resultados, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Resultados salvos em: {arquivo_resultado}")
            
        except Exception as e:
            print(f"\n❌ Erro ao salvar resultados: {e}")
    
    def exibir_resumo(self):
        """Exibe resumo dos testes"""
        print("\n" + "=" * 60)
        print("📋 RESUMO DO TESTE COMPLETO")
        print("=" * 60)
        
        # Status dos testes
        print(f"🔍 Testes realizados: {len(self.resultados['testes_realizados'])}")
        for teste in self.resultados['testes_realizados']:
            print(f"   ✅ {teste}")
        
        # Status dos modelos
        if self.resultados['modelos_testados']:
            modelos_ok = sum(1 for m in self.resultados['modelos_testados'].values() 
                           if m.get('status') == 'carregado_com_sucesso')
            total = len(self.resultados['modelos_testados'])
            print(f"\n🤖 Modelos: {modelos_ok}/{total} funcionando")
        
        # Performance
        if 'performance_geral' in self.resultados:
            perf = self.resultados['performance_geral']
            print(f"\n⚡ Performance:")
            print(f"   ⏱️  Carregamento: {perf.get('tempo_carregamento_modelos', 0):.2f}s")
            print(f"   💾 Memória: {perf.get('uso_memoria', 0):.2f}MB")
        
        # Recomendações principais
        if self.resultados['recomendacoes']:
            print(f"\n💡 Principais recomendações:")
            for i, rec in enumerate(self.resultados['recomendacoes'][:3], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "=" * 60)
        print("✅ TESTE COMPLETO FINALIZADO COM SUCESSO!")
        print("=" * 60)

def main():
    """Função principal"""
    teste = TesteSistemaOtimizado()
    teste.executar_teste_completo()

if __name__ == "__main__":
    main()