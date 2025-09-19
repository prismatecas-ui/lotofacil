#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste do Sistema de Predição Integrada
Testa a integração do modelo TensorFlow com 66 features no sistema de jogadas
"""

import sys
import os
import json
from datetime import datetime

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jogar import SistemaLotofacil
from dados.dados import setup_logger

def teste_carregamento_sistema():
    """Testa o carregamento do sistema integrado"""
    print("\n🔧 TESTE 1: Carregamento do Sistema")
    print("-" * 50)
    
    try:
        sistema = SistemaLotofacil()
        print("✅ Sistema principal carregado com sucesso")
        
        # Verificar se o sistema de predição foi inicializado
        if hasattr(sistema, 'predicao_integrada'):
            print("✅ Sistema de predição integrada inicializado")
        else:
            print("❌ Sistema de predição integrada NÃO encontrado")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Erro no carregamento: {e}")
        return False

def teste_carregamento_dados():
    """Testa o carregamento de dados históricos"""
    print("\n📊 TESTE 2: Carregamento de Dados")
    print("-" * 50)
    
    try:
        sistema = SistemaLotofacil()
        dados = sistema.carregar_dados()
        
        if dados is not None and len(dados) > 0:
            print(f"✅ Dados carregados: {len(dados)} registros")
            print(f"📅 Período: {dados['concurso'].min()} a {dados['concurso'].max()}")
            return True
        else:
            print("❌ Nenhum dado histórico encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Erro no carregamento de dados: {e}")
        return False

def teste_predicao_individual():
    """Testa predição para um jogo específico"""
    print("\n🎯 TESTE 3: Predição Individual")
    print("-" * 50)
    
    try:
        sistema = SistemaLotofacil()
        dados = sistema.carregar_dados()
        
        if dados is None:
            print("⚠️ Dados não disponíveis, testando com fallback")
        
        # Jogo de teste
        jogo_teste = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        
        print(f"🎲 Jogo teste: {jogo_teste}")
        
        # Fazer predição
        probabilidade = sistema.fazer_predicao(jogo_teste, dados)
        
        print(f"📈 Probabilidade calculada: {probabilidade:.2f}%")
        
        if 0 <= probabilidade <= 100:
            print("✅ Predição realizada com sucesso")
            return True
        else:
            print(f"❌ Probabilidade inválida: {probabilidade}")
            return False
            
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return False

def teste_geracao_jogos():
    """Testa a geração de jogos inteligentes"""
    print("\n🎮 TESTE 4: Geração de Jogos Inteligentes")
    print("-" * 50)
    
    try:
        sistema = SistemaLotofacil()
        
        # Gerar 3 jogos para teste
        jogos = sistema.gerar_jogos_inteligentes(3)
        
        if len(jogos) == 3:
            print(f"✅ {len(jogos)} jogos gerados com sucesso")
            
            for i, jogo in enumerate(jogos, 1):
                numeros = jogo['numeros']
                prob = jogo['probabilidade']
                tipo = jogo.get('tipo', 'desconhecido')
                
                print(f"\n🎲 Jogo {i}: {numeros}")
                print(f"📈 Probabilidade: {prob:.2f}%")
                print(f"🔧 Tipo: {tipo}")
                
                # Validações básicas
                if len(numeros) != 15:
                    print(f"❌ Jogo inválido: {len(numeros)} números")
                    return False
                    
                if not all(1 <= n <= 25 for n in numeros):
                    print("❌ Números fora do range válido")
                    return False
                    
                if len(set(numeros)) != 15:
                    print("❌ Números duplicados encontrados")
                    return False
            
            return True
        else:
            print(f"❌ Número incorreto de jogos: {len(jogos)}")
            return False
            
    except Exception as e:
        print(f"❌ Erro na geração de jogos: {e}")
        return False

def teste_validacao_jogos():
    """Testa o sistema de validação de jogos"""
    print("\n✅ TESTE 5: Validação de Jogos")
    print("-" * 50)
    
    try:
        sistema = SistemaLotofacil()
        
        # Teste com jogo válido
        jogo_valido = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 2, 4, 6, 8, 10]
        resultado_valido = sistema._validar_jogo(jogo_valido, 80.0)
        print(f"🎲 Jogo válido: {resultado_valido}")
        
        # Teste com jogo inválido (muitos consecutivos)
        jogo_invalido = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        resultado_invalido = sistema._validar_jogo(jogo_invalido, 80.0)
        print(f"🎲 Jogo inválido: {resultado_invalido}")
        
        # Teste com probabilidade baixa
        jogo_prob_baixa = sistema._validar_jogo(jogo_valido, 50.0)
        print(f"🎲 Probabilidade baixa: {jogo_prob_baixa}")
        
        if resultado_valido and not resultado_invalido and not jogo_prob_baixa:
            print("✅ Sistema de validação funcionando corretamente")
            return True
        else:
            print("❌ Sistema de validação com problemas")
            return False
            
    except Exception as e:
        print(f"❌ Erro na validação: {e}")
        return False

def executar_todos_testes():
    """Executa todos os testes do sistema"""
    logger = setup_logger('teste_predicao')
    
    print("\n" + "=" * 70)
    print("🧪 TESTE COMPLETO DO SISTEMA DE PREDIÇÃO INTEGRADA")
    print("=" * 70)
    
    testes = [
        ("Carregamento do Sistema", teste_carregamento_sistema),
        ("Carregamento de Dados", teste_carregamento_dados),
        ("Predição Individual", teste_predicao_individual),
        ("Geração de Jogos", teste_geracao_jogos),
        ("Validação de Jogos", teste_validacao_jogos)
    ]
    
    resultados = []
    
    for nome, teste_func in testes:
        try:
            resultado = teste_func()
            resultados.append((nome, resultado))
            
            if resultado:
                logger.info(f"Teste '{nome}' passou")
            else:
                logger.error(f"Teste '{nome}' falhou")
                
        except Exception as e:
            logger.error(f"Erro no teste '{nome}': {e}")
            resultados.append((nome, False))
    
    # Relatório final
    print("\n" + "=" * 70)
    print("📋 RELATÓRIO FINAL DOS TESTES")
    print("=" * 70)
    
    testes_passaram = 0
    for nome, resultado in resultados:
        status = "✅ PASSOU" if resultado else "❌ FALHOU"
        print(f"{nome:<25} {status}")
        if resultado:
            testes_passaram += 1
    
    print(f"\n📊 Resultado: {testes_passaram}/{len(resultados)} testes passaram")
    
    if testes_passaram == len(resultados):
        print("🎉 TODOS OS TESTES PASSARAM! Sistema integrado com sucesso.")
    else:
        print("⚠️ Alguns testes falharam. Verifique os logs para detalhes.")
    
    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    relatorio = {
        'timestamp': timestamp,
        'testes_executados': len(resultados),
        'testes_passaram': testes_passaram,
        'resultados': [{'teste': nome, 'passou': resultado} for nome, resultado in resultados]
    }
    
    os.makedirs("resultados", exist_ok=True)
    arquivo_relatorio = f"resultados/teste_predicao_{timestamp}.json"
    
    with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Relatório salvo em: {arquivo_relatorio}")
    
    return testes_passaram == len(resultados)

if __name__ == "__main__":
    sucesso = executar_todos_testes()
    sys.exit(0 if sucesso else 1)