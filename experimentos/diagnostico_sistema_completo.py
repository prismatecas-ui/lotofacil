#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnóstico Completo do Sistema de IA Lotofácil
Analisa performance, limitações e propõe otimizações
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
except ImportError as e:
    print(f"Erro ao importar bibliotecas: {e}")
    sys.exit(1)

class DiagnosticoSistema:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.resultados_dir = self.base_dir / "experimentos" / "resultados"
        self.modelo_dir = self.base_dir / "modelo"
        self.dados_dir = self.base_dir / "base"
        
        # Criar diretório de resultados se não existir
        self.resultados_dir.mkdir(exist_ok=True)
        
        self.diagnostico = {
            "timestamp": datetime.now().isoformat(),
            "analise_dados": {},
            "analise_modelos": {},
            "limitacoes_identificadas": [],
            "recomendacoes_otimizacao": [],
            "metricas_performance": {},
            "score_geral": 0.0
        }
    
    def analisar_dados_treinamento(self):
        """Analisa a qualidade dos dados de treinamento"""
        print("\n=== ANÁLISE DOS DADOS DE TREINAMENTO ===")
        
        try:
            # Verificar dataset completo
            dataset_path = self.base_dir / "experimentos" / "datasets" / "dataset_lotofacil_completo_20250919_080901.csv"
            if dataset_path.exists():
                df = pd.read_csv(dataset_path)
                print(f"Dataset encontrado: {len(df)} amostras, {len(df.columns)} features")
                
                # Análise de qualidade dos dados
                missing_values = df.isnull().sum().sum()
                duplicates = df.duplicated().sum()
                
                # Análise de distribuição das features
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_stats = {
                    "total_features": len(df.columns),
                    "numeric_features": len(numeric_cols),
                    "missing_values": int(missing_values),
                    "duplicates": int(duplicates),
                    "data_quality_score": max(0, 100 - (missing_values/len(df) * 100) - (duplicates/len(df) * 10))
                }
                
                # Análise de correlação
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    high_corr = (corr_matrix.abs() > 0.9).sum().sum() - len(corr_matrix)
                    feature_stats["high_correlations"] = int(high_corr)
                
                self.diagnostico["analise_dados"] = feature_stats
                print(f"Qualidade dos dados: {feature_stats['data_quality_score']:.1f}%")
                
                if missing_values > 0:
                    self.diagnostico["limitacoes_identificadas"].append({
                        "tipo": "Dados Faltantes",
                        "descricao": f"{missing_values} valores faltantes encontrados",
                        "severidade": "Alta" if missing_values > len(df) * 0.05 else "Média"
                    })
                
                if duplicates > 0:
                    self.diagnostico["limitacoes_identificadas"].append({
                        "tipo": "Dados Duplicados",
                        "descricao": f"{duplicates} registros duplicados",
                        "severidade": "Média"
                    })
                
                return df
            else:
                print("Dataset não encontrado!")
                return None
                
        except Exception as e:
            print(f"Erro na análise de dados: {e}")
            return None
    
    def analisar_modelos_salvos(self):
        """Analisa os modelos salvos e sua performance"""
        print("\n=== ANÁLISE DOS MODELOS SALVOS ===")
        
        modelos_encontrados = []
        
        # Verificar modelos TensorFlow
        saved_models_dir = self.modelo_dir / "saved_models"
        if saved_models_dir.exists():
            for model_dir in saved_models_dir.iterdir():
                if model_dir.is_dir():
                    model_path = model_dir / "modelo.h5"
                    if model_path.exists():
                        try:
                            modelo = tf.keras.models.load_model(str(model_path))
                            info_modelo = {
                                "nome": model_dir.name,
                                "tipo": "TensorFlow",
                                "camadas": len(modelo.layers),
                                "parametros": modelo.count_params(),
                                "tamanho_mb": model_path.stat().st_size / (1024*1024)
                            }
                            modelos_encontrados.append(info_modelo)
                            print(f"Modelo TF encontrado: {info_modelo['nome']} ({info_modelo['parametros']} parâmetros)")
                        except Exception as e:
                            print(f"Erro ao carregar modelo {model_dir.name}: {e}")
        
        # Verificar checkpoints
        checkpoints_dir = self.modelo_dir / "checkpoints"
        if checkpoints_dir.exists():
            for checkpoint_dir in checkpoints_dir.iterdir():
                if checkpoint_dir.is_dir():
                    checkpoint_path = checkpoint_dir / "best_model.h5"
                    if checkpoint_path.exists():
                        print(f"Checkpoint encontrado: {checkpoint_dir.name}")
        
        self.diagnostico["analise_modelos"]["modelos_encontrados"] = modelos_encontrados
        
        if not modelos_encontrados:
            self.diagnostico["limitacoes_identificadas"].append({
                "tipo": "Modelos Ausentes",
                "descricao": "Nenhum modelo treinado encontrado",
                "severidade": "Crítica"
            })
    
    def avaliar_metricas_performance(self):
        """Avalia as métricas de performance existentes"""
        print("\n=== ANÁLISE DE MÉTRICAS DE PERFORMANCE ===")
        
        # Carregar métricas existentes
        metricas_files = list(self.resultados_dir.glob("metricas_*.json"))
        
        if metricas_files:
            for metrics_file in metricas_files:
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metricas = json.load(f)
                    
                    print(f"\nMétricas de {metricas.get('model_name', 'Modelo Desconhecido')}:")
                    
                    if 'basic_metrics' in metricas:
                        basic = metricas['basic_metrics']
                        print(f"  Accuracy: {basic.get('accuracy', 0):.3f}")
                        print(f"  Precision: {basic.get('precision', 0):.3f}")
                        print(f"  Recall: {basic.get('recall', 0):.3f}")
                        print(f"  F1-Score: {basic.get('f1_score', 0):.3f}")
                    
                    if 'hit_rates' in metricas:
                        hit_rates = metricas['hit_rates']
                        print(f"  Taxa de acerto 15 números: {hit_rates.get('hit_rate_15', 0):.3f}")
                        print(f"  Taxa de acerto 14 números: {hit_rates.get('hit_rate_14', 0):.3f}")
                        print(f"  Taxa de acerto 13 números: {hit_rates.get('hit_rate_13', 0):.3f}")
                    
                    if 'overall_score' in metricas:
                        score = metricas['overall_score']
                        print(f"  Score Geral: {score:.3f}")
                        
                        if score < 0.5:
                            self.diagnostico["limitacoes_identificadas"].append({
                                "tipo": "Performance Baixa",
                                "descricao": f"Score geral de {score:.3f} indica performance inadequada",
                                "severidade": "Alta"
                            })
                    
                    self.diagnostico["metricas_performance"][metrics_file.stem] = metricas
                    
                except Exception as e:
                    print(f"Erro ao carregar métricas de {metrics_file}: {e}")
        else:
            print("Nenhuma métrica de performance encontrada")
            self.diagnostico["limitacoes_identificadas"].append({
                "tipo": "Métricas Ausentes",
                "descricao": "Nenhuma métrica de performance disponível",
                "severidade": "Alta"
            })
    
    def identificar_limitacoes_arquitetura(self):
        """Identifica limitações na arquitetura atual"""
        print("\n=== ANÁLISE DE LIMITAÇÕES DA ARQUITETURA ===")
        
        # Carregar análise de limitações existente
        limitacoes_file = self.resultados_dir / "dados_limitacoes_modelo_20250918_144845.json"
        if limitacoes_file.exists():
            try:
                with open(limitacoes_file, 'r', encoding='utf-8') as f:
                    limitacoes_existentes = json.load(f)
                
                # Processar limitações identificadas
                for categoria, dados in limitacoes_existentes.items():
                    if isinstance(dados, dict) and 'limitations' in dados:
                        for limitacao in dados['limitations']:
                            self.diagnostico["limitacoes_identificadas"].append({
                                "tipo": limitacao.get('type', 'Limitação Desconhecida'),
                                "descricao": limitacao.get('description', ''),
                                "severidade": limitacao.get('severity', 'Média'),
                                "categoria": categoria
                            })
                
                print("Limitações arquiteturais identificadas:")
                for limitacao in self.diagnostico["limitacoes_identificadas"]:
                    if 'categoria' in limitacao:
                        print(f"  - {limitacao['tipo']}: {limitacao['descricao']}")
                        
            except Exception as e:
                print(f"Erro ao carregar limitações: {e}")
    
    def gerar_recomendacoes_otimizacao(self):
        """Gera recomendações específicas para otimização"""
        print("\n=== GERAÇÃO DE RECOMENDAÇÕES DE OTIMIZAÇÃO ===")
        
        recomendacoes = []
        
        # Recomendações baseadas nas limitações identificadas
        for limitacao in self.diagnostico["limitacoes_identificadas"]:
            if "Performance Baixa" in limitacao["tipo"]:
                recomendacoes.extend([
                    {
                        "categoria": "Arquitetura",
                        "titulo": "Implementar Ensemble de Modelos",
                        "descricao": "Combinar múltiplos algoritmos (Random Forest, XGBoost, Neural Networks) para melhorar precisão",
                        "impacto_esperado": "15-25% melhoria na taxa de acerto",
                        "prioridade": "Alta"
                    },
                    {
                        "categoria": "Features",
                        "titulo": "Engenharia de Features Avançada",
                        "descricao": "Adicionar features temporais, padrões de sequência e análise de tendências",
                        "impacto_esperado": "10-15% melhoria na precisão",
                        "prioridade": "Alta"
                    }
                ])
            
            if "Validação" in limitacao["tipo"]:
                recomendacoes.append({
                    "categoria": "Validação",
                    "titulo": "Implementar Validação Cruzada Temporal",
                    "descricao": "Usar validação com janela deslizante respeitando ordem cronológica",
                    "impacto_esperado": "Estimativa mais confiável da performance real",
                    "prioridade": "Alta"
                })
            
            if "Representação" in limitacao["tipo"]:
                recomendacoes.append({
                    "categoria": "Dados",
                    "titulo": "Melhorar Representação dos Dados",
                    "descricao": "Implementar embeddings e features numéricas além da codificação binária",
                    "impacto_esperado": "5-10% melhoria na capacidade preditiva",
                    "prioridade": "Média"
                })
        
        # Recomendações específicas para atingir 85-90% de acerto
        recomendacoes.extend([
            {
                "categoria": "Algoritmo",
                "titulo": "Implementar Gradient Boosting Otimizado",
                "descricao": "Usar XGBoost ou LightGBM com hyperparameter tuning automático",
                "impacto_esperado": "20-30% melhoria na taxa de acerto",
                "prioridade": "Crítica"
            },
            {
                "categoria": "Dados",
                "titulo": "Análise de Padrões Temporais Profunda",
                "descricao": "Implementar LSTM para capturar dependências temporais de longo prazo",
                "impacto_esperado": "15-20% melhoria na precisão",
                "prioridade": "Alta"
            },
            {
                "categoria": "Otimização",
                "titulo": "Hyperparameter Tuning Automático",
                "descricao": "Usar Optuna ou Hyperopt para otimização automática de hiperparâmetros",
                "impacto_esperado": "5-10% melhoria na performance",
                "prioridade": "Média"
            },
            {
                "categoria": "Ensemble",
                "titulo": "Stacking de Modelos Heterogêneos",
                "descricao": "Combinar diferentes tipos de modelos com meta-learner",
                "impacto_esperado": "10-15% melhoria na robustez",
                "prioridade": "Alta"
            }
        ])
        
        self.diagnostico["recomendacoes_otimizacao"] = recomendacoes
        
        print("Recomendações geradas:")
        for rec in recomendacoes:
            print(f"  [{rec['prioridade']}] {rec['titulo']}: {rec['impacto_esperado']}")
    
    def calcular_score_geral(self):
        """Calcula um score geral do sistema"""
        score = 100.0
        
        # Penalizar por limitações
        for limitacao in self.diagnostico["limitacoes_identificadas"]:
            if limitacao["severidade"] == "Crítica":
                score -= 30
            elif limitacao["severidade"] == "Alta":
                score -= 20
            elif limitacao["severidade"] == "Média":
                score -= 10
        
        # Bonificar por qualidade dos dados
        if "analise_dados" in self.diagnostico and "data_quality_score" in self.diagnostico["analise_dados"]:
            data_score = self.diagnostico["analise_dados"]["data_quality_score"]
            score = score * (data_score / 100)
        
        # Bonificar por modelos encontrados
        if "analise_modelos" in self.diagnostico and "modelos_encontrados" in self.diagnostico["analise_modelos"]:
            num_modelos = len(self.diagnostico["analise_modelos"]["modelos_encontrados"])
            if num_modelos > 0:
                score += min(10, num_modelos * 5)
        
        self.diagnostico["score_geral"] = max(0, min(100, score))
        
        print(f"\nScore Geral do Sistema: {self.diagnostico['score_geral']:.1f}/100")
    
    def executar_diagnostico_completo(self):
        """Executa o diagnóstico completo do sistema"""
        print("\n" + "="*60)
        print("    DIAGNÓSTICO COMPLETO DO SISTEMA LOTOFÁCIL")
        print("="*60)
        
        # Executar todas as análises
        dados = self.analisar_dados_treinamento()
        self.analisar_modelos_salvos()
        self.avaliar_metricas_performance()
        self.identificar_limitacoes_arquitetura()
        self.gerar_recomendacoes_otimizacao()
        self.calcular_score_geral()
        
        # Salvar diagnóstico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        diagnostico_file = self.resultados_dir / f"diagnostico_completo_{timestamp}.json"
        
        with open(diagnostico_file, 'w', encoding='utf-8') as f:
            json.dump(self.diagnostico, f, indent=2, ensure_ascii=False)
        
        print(f"\nDiagnóstico salvo em: {diagnostico_file}")
        
        # Gerar relatório resumido
        self.gerar_relatorio_resumido()
        
        return self.diagnostico
    
    def gerar_relatorio_resumido(self):
        """Gera um relatório resumido do diagnóstico"""
        print("\n" + "="*60)
        print("                RELATÓRIO RESUMIDO")
        print("="*60)
        
        print(f"\n🔍 ANÁLISE GERAL:")
        print(f"   Score do Sistema: {self.diagnostico['score_geral']:.1f}/100")
        print(f"   Limitações Identificadas: {len(self.diagnostico['limitacoes_identificadas'])}")
        print(f"   Recomendações Geradas: {len(self.diagnostico['recomendacoes_otimizacao'])}")
        
        print(f"\n⚠️  PRINCIPAIS LIMITAÇÕES:")
        limitacoes_criticas = [l for l in self.diagnostico['limitacoes_identificadas'] if l['severidade'] in ['Crítica', 'Alta']]
        for i, limitacao in enumerate(limitacoes_criticas[:5], 1):
            print(f"   {i}. [{limitacao['severidade']}] {limitacao['tipo']}: {limitacao['descricao']}")
        
        print(f"\n🚀 RECOMENDAÇÕES PRIORITÁRIAS:")
        recomendacoes_prioritarias = [r for r in self.diagnostico['recomendacoes_otimizacao'] if r['prioridade'] in ['Crítica', 'Alta']]
        for i, rec in enumerate(recomendacoes_prioritarias[:5], 1):
            print(f"   {i}. [{rec['prioridade']}] {rec['titulo']}")
            print(f"      Impacto: {rec['impacto_esperado']}")
        
        print(f"\n📊 PRÓXIMOS PASSOS PARA ATINGIR 85-90% DE ACERTO:")
        print(f"   1. Implementar ensemble de modelos (XGBoost + LSTM + Random Forest)")
        print(f"   2. Melhorar engenharia de features com padrões temporais")
        print(f"   3. Aplicar validação cruzada temporal")
        print(f"   4. Otimizar hiperparâmetros automaticamente")
        print(f"   5. Implementar stacking de modelos heterogêneos")
        
        print("\n" + "="*60)

def main():
    """Função principal"""
    diagnostico = DiagnosticoSistema()
    resultado = diagnostico.executar_diagnostico_completo()
    
    print("\n✅ Diagnóstico completo finalizado!")
    print("\n📋 Para implementar as otimizações, execute:")
    print("   python experimentos/implementar_otimizacoes.py")
    
    return resultado

if __name__ == "__main__":
    main()