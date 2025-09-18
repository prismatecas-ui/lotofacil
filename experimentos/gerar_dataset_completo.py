import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging
from typing import Dict, Tuple

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos locais
from experimentos.feature_engineering import FeatureEngineeringLotofacil
from experimentos.feature_selector import FeatureSelectorLotofacil
from experimentos.preprocessor import PreprocessadorAvancadoLotofacil
from dados.dados import carregar_dados, preparar_dados

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experimentos/logs/dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeradorDatasetCompleto:
    """Gerador de dataset completo com todas as features avançadas"""
    
    def __init__(self, guia_dados: str = 'Importar_Ciclo'):
        self.guia_dados = guia_dados
        self.feature_engineer = FeatureEngineeringLotofacil()
        self.feature_selector = FeatureSelectorLotofacil()
        self.preprocessor = PreprocessadorAvancadoLotofacil()
        
        # Criar diretórios necessários
        os.makedirs('experimentos/datasets', exist_ok=True)
        os.makedirs('experimentos/logs', exist_ok=True)
        os.makedirs('experimentos/modelos', exist_ok=True)
        
    def carregar_dados_historicos(self) -> pd.DataFrame:
        """Carrega e prepara dados históricos"""
        logger.info("Carregando dados históricos...")
        
        try:
            # Carregar dados usando função existente
            dados_brutos = carregar_dados(self.guia_dados)
            logger.info(f"Dados carregados: {len(dados_brutos)} registros")
            
            return dados_brutos
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise
    
    def gerar_features_estatisticas(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Gera todas as features estatísticas"""
        logger.info("Gerando features estatísticas...")
        
        try:
            features_estat = self.feature_engineer.criar_features_estatisticas(dados)
            logger.info(f"Features estatísticas geradas: {features_estat.shape[1]} colunas")
            return features_estat
            
        except Exception as e:
            logger.error(f"Erro ao gerar features estatísticas: {e}")
            raise
    
    def gerar_features_temporais(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Gera todas as features temporais"""
        logger.info("Gerando features temporais...")
        
        try:
            features_temp = self.feature_engineer.criar_features_temporais(dados)
            logger.info(f"Features temporais geradas: {features_temp.shape[1]} colunas")
            return features_temp
            
        except Exception as e:
            logger.error(f"Erro ao gerar features temporais: {e}")
            raise
    
    def gerar_features_padroes(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Gera features de padrões para cada concurso"""
        logger.info("Gerando features de padrões...")
        
        try:
            features_padroes_list = []
            
            for idx in range(len(dados)):
                # Usar dados até o concurso atual para análise de padrões
                dados_ate_atual = dados.iloc[:idx+1]
                
                if len(dados_ate_atual) > 0:
                    features_padroes = self.feature_engineer.criar_features_padroes(dados_ate_atual)
                    
                    # Adicionar identificador do concurso
                    if hasattr(dados.iloc[idx], 'Concurso'):
                        features_padroes['concurso'] = dados.iloc[idx]['Concurso']
                    else:
                        features_padroes['concurso'] = idx + 1
                    
                    features_padroes_list.append(features_padroes)
            
            # Converter para DataFrame
            if features_padroes_list:
                features_padroes_df = pd.DataFrame(features_padroes_list)
                logger.info(f"Features de padrões geradas: {features_padroes_df.shape[1]} colunas")
                return features_padroes_df
            else:
                logger.warning("Nenhuma feature de padrão foi gerada")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Erro ao gerar features de padrões: {e}")
            raise
    
    def combinar_todas_features(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Combina todas as features em um único dataset"""
        logger.info("Combinando todas as features...")
        
        try:
            # Gerar cada tipo de feature
            features_estat = self.gerar_features_estatisticas(dados)
            features_temp = self.gerar_features_temporais(dados)
            features_padroes = self.gerar_features_padroes(dados)
            
            # Combinar features
            dataset_completo = features_estat.copy()
            
            # Merge com features temporais
            if not features_temp.empty and 'concurso' in features_temp.columns:
                dataset_completo = pd.merge(
                    dataset_completo, features_temp, 
                    on='concurso', how='inner', suffixes=('', '_temp')
                )
            
            # Merge com features de padrões
            if not features_padroes.empty and 'concurso' in features_padroes.columns:
                dataset_completo = pd.merge(
                    dataset_completo, features_padroes, 
                    on='concurso', how='inner', suffixes=('', '_padr')
                )
            
            logger.info(f"Dataset completo: {dataset_completo.shape}")
            return dataset_completo
            
        except Exception as e:
            logger.error(f"Erro ao combinar features: {e}")
            raise
    
    def criar_targets(self, dados: pd.DataFrame, dataset_features: pd.DataFrame) -> pd.DataFrame:
        """Cria variáveis target para predição"""
        logger.info("Criando variáveis target...")
        
        try:
            targets = pd.DataFrame()
            
            # Target 1: Soma dos números do próximo concurso
            if 'soma_total' in dataset_features.columns:
                targets['target_soma'] = dataset_features['soma_total'].shift(-1)
            
            # Target 2: Quantidade de pares do próximo concurso
            if 'qtd_pares' in dataset_features.columns:
                targets['target_pares'] = dataset_features['qtd_pares'].shift(-1)
            
            # Target 3: Quantidade de números na primeira metade (1-12)
            if 'primeira_metade' in dataset_features.columns:
                targets['target_primeira_metade'] = dataset_features['primeira_metade'].shift(-1)
            
            # Target 4: Máximo gap do próximo concurso
            if 'gap_max' in dataset_features.columns:
                targets['target_gap_max'] = dataset_features['gap_max'].shift(-1)
            
            # Adicionar concurso para merge
            if 'concurso' in dataset_features.columns:
                targets['concurso'] = dataset_features['concurso']
            
            # Remover última linha (sem target)
            targets = targets.iloc[:-1]
            
            logger.info(f"Targets criados: {targets.shape}")
            return targets
            
        except Exception as e:
            logger.error(f"Erro ao criar targets: {e}")
            raise
    
    def preprocessar_dataset(self, dataset: pd.DataFrame, 
                           usar_selecao_features: bool = True,
                           n_features: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """Preprocessa o dataset completo"""
        logger.info("Preprocessando dataset...")
        
        try:
            # Separar features numéricas
            numeric_columns = dataset.select_dtypes(include=[np.number]).columns
            dataset_numeric = dataset[numeric_columns].copy()
            
            # Remover colunas com muitos NaN
            threshold_nan = 0.5  # 50% de valores NaN
            cols_to_keep = []
            for col in dataset_numeric.columns:
                if dataset_numeric[col].isnull().mean() < threshold_nan:
                    cols_to_keep.append(col)
            
            dataset_clean = dataset_numeric[cols_to_keep].copy()
            logger.info(f"Colunas mantidas após limpeza: {len(cols_to_keep)}")
            
            # Análise de distribuições
            stats = self.preprocessor.analisar_distribuicoes(dataset_clean)
            
            # Preprocessamento automático
            dataset_preprocessed = self.preprocessor.preprocessar_automatico(dataset_clean)
            
            # Seleção de features (se solicitado)
            if usar_selecao_features and len(dataset_preprocessed.columns) > n_features:
                # Criar target simples para seleção (usar primeira coluna numérica)
                target_col = dataset_preprocessed.columns[0]
                target = dataset_preprocessed[target_col]
                features = dataset_preprocessed.drop(columns=[target_col])
                
                # Selecionar melhores features
                features_selected, feature_info = self.feature_selector.selecionar_melhores_features(
                    features, target, n_features
                )
                
                # Adicionar target de volta
                dataset_final = pd.concat([features_selected, target], axis=1)
            else:
                dataset_final = dataset_preprocessed
            
            # Gerar relatório
            relatorio = {
                'shape_original': dataset.shape,
                'shape_final': dataset_final.shape,
                'features_removidas': len(dataset.columns) - len(dataset_final.columns),
                'stats_distribuicoes': stats,
                'preprocessamento': self.preprocessor.gerar_relatorio_preprocessamento(
                    dataset_clean, dataset_final
                )
            }
            
            logger.info(f"Dataset preprocessado: {dataset_final.shape}")
            return dataset_final, relatorio
            
        except Exception as e:
            logger.error(f"Erro no preprocessamento: {e}")
            raise
    
    def salvar_dataset(self, dataset: pd.DataFrame, targets: pd.DataFrame, 
                      relatorio: Dict, nome_base: str = 'dataset_lotofacil_completo'):
        """Salva o dataset e metadados"""
        logger.info("Salvando dataset...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Salvar dataset principal
            dataset_path = f'experimentos/datasets/{nome_base}_{timestamp}.csv'
            dataset.to_csv(dataset_path, index=False)
            logger.info(f"Dataset salvo: {dataset_path}")
            
            # Salvar targets
            targets_path = f'experimentos/datasets/{nome_base}_targets_{timestamp}.csv'
            targets.to_csv(targets_path, index=False)
            logger.info(f"Targets salvos: {targets_path}")
            
            # Salvar relatório
            relatorio_path = f'experimentos/datasets/{nome_base}_relatorio_{timestamp}.txt'
            with open(relatorio_path, 'w', encoding='utf-8') as f:
                f.write(f"Relatório de Geração do Dataset - {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Shape do dataset final: {dataset.shape}\n")
                f.write(f"Shape dos targets: {targets.shape}\n")
                f.write(f"Features no dataset: {list(dataset.columns)}\n\n")
                
                if 'preprocessamento' in relatorio:
                    f.write("Preprocessamento aplicado:\n")
                    for key, value in relatorio['preprocessamento'].items():
                        f.write(f"  {key}: {value}\n")
            
            logger.info(f"Relatório salvo: {relatorio_path}")
            
            # Salvar preprocessadores
            self.preprocessor.salvar_preprocessadores('experimentos/modelos/preprocessadores')
            
            return {
                'dataset_path': dataset_path,
                'targets_path': targets_path,
                'relatorio_path': relatorio_path
            }
            
        except Exception as e:
            logger.error(f"Erro ao salvar dataset: {e}")
            raise
    
    def executar_pipeline_completo(self, 
                                 usar_selecao_features: bool = True,
                                 n_features: int = 100) -> Dict:
        """Executa o pipeline completo de geração do dataset"""
        logger.info("Iniciando pipeline completo de geração do dataset...")
        
        try:
            # 1. Carregar dados
            dados_historicos = self.carregar_dados_historicos()
            
            # 2. Gerar todas as features
            dataset_features = self.combinar_todas_features(dados_historicos)
            
            # 3. Criar targets
            targets = self.criar_targets(dados_historicos, dataset_features)
            
            # 4. Preprocessar
            dataset_final, relatorio = self.preprocessar_dataset(
                dataset_features, usar_selecao_features, n_features
            )
            
            # 5. Salvar tudo
            paths = self.salvar_dataset(dataset_final, targets, relatorio)
            
            logger.info("Pipeline completo executado com sucesso!")
            
            return {
                'sucesso': True,
                'dataset_shape': dataset_final.shape,
                'targets_shape': targets.shape,
                'paths': paths,
                'relatorio': relatorio
            }
            
        except Exception as e:
            logger.error(f"Erro no pipeline completo: {e}")
            return {
                'sucesso': False,
                'erro': str(e)
            }

def main():
    """Função principal para execução do script"""
    print("Iniciando geração do dataset completo da Lotofácil...")
    
    # Criar gerador
    gerador = GeradorDatasetCompleto()
    
    # Executar pipeline
    resultado = gerador.executar_pipeline_completo(
        usar_selecao_features=True,
        n_features=80  # Selecionar 80 melhores features
    )
    
    if resultado['sucesso']:
        print(f"\n✅ Dataset gerado com sucesso!")
        print(f"📊 Shape do dataset: {resultado['dataset_shape']}")
        print(f"🎯 Shape dos targets: {resultado['targets_shape']}")
        print(f"📁 Arquivos salvos em: experimentos/datasets/")
    else:
        print(f"\n❌ Erro na geração: {resultado['erro']}")

if __name__ == "__main__":
    main()