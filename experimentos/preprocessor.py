import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    QuantileTransformer, PowerTransformer, Normalizer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessadorAvancadoLotofacil:
    """Preprocessador avançado para features da Lotofácil"""
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'quantile_uniform': QuantileTransformer(output_distribution='uniform'),
            'quantile_normal': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson'),
            'normalizer': Normalizer()
        }
        
        self.imputers = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent'),
            'constant': SimpleImputer(strategy='constant', fill_value=0),
            'knn': KNNImputer(n_neighbors=5)
        }
        
        self.fitted_scalers = {}
        self.fitted_imputers = {}
        self.feature_stats = {}
        
    def analisar_distribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Analisa distribuições das features para escolher melhor preprocessamento"""
        logger.info("Analisando distribuições das features...")
        
        stats = []
        
        for col in X.columns:
            serie = X[col].dropna()
            
            if len(serie) == 0:
                continue
                
            stat = {
                'feature': col,
                'mean': serie.mean(),
                'median': serie.median(),
                'std': serie.std(),
                'min': serie.min(),
                'max': serie.max(),
                'skewness': serie.skew(),
                'kurtosis': serie.kurtosis(),
                'missing_pct': X[col].isnull().mean() * 100,
                'zeros_pct': (serie == 0).mean() * 100,
                'unique_values': serie.nunique(),
                'outliers_pct': self._calcular_outliers_pct(serie)
            }
            
            # Recomendar preprocessamento
            stat['recomendacao_scaler'] = self._recomendar_scaler(stat)
            stat['recomendacao_imputer'] = self._recomendar_imputer(stat)
            
            stats.append(stat)
        
        self.feature_stats = pd.DataFrame(stats)
        return self.feature_stats
    
    def _calcular_outliers_pct(self, serie: pd.Series) -> float:
        """Calcula percentual de outliers usando IQR"""
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (serie < lower_bound) | (serie > upper_bound)
        return outliers.mean() * 100
    
    def _recomendar_scaler(self, stats: Dict) -> str:
        """Recomenda o melhor scaler baseado nas estatísticas"""
        # Se há muitos outliers, usar RobustScaler
        if stats['outliers_pct'] > 10:
            return 'robust'
        
        # Se distribuição é muito assimétrica, usar QuantileTransformer
        if abs(stats['skewness']) > 2:
            return 'quantile_normal'
        
        # Se há valores negativos e queremos normalizar, usar PowerTransformer
        if stats['min'] < 0 and abs(stats['skewness']) > 1:
            return 'power'
        
        # Caso padrão: StandardScaler
        return 'standard'
    
    def _recomendar_imputer(self, stats: Dict) -> str:
        """Recomenda o melhor imputer baseado nas estatísticas"""
        # Se poucos valores missing, usar KNN
        if stats['missing_pct'] < 5 and stats['missing_pct'] > 0:
            return 'knn'
        
        # Se distribuição é assimétrica, usar mediana
        if abs(stats['skewness']) > 1:
            return 'median'
        
        # Se há muitos zeros, usar most_frequent
        if stats['zeros_pct'] > 20:
            return 'most_frequent'
        
        # Caso padrão: média
        return 'mean'
    
    def preprocessar_automatico(self, X: pd.DataFrame, 
                              usar_recomendacoes: bool = True) -> pd.DataFrame:
        """Preprocessa automaticamente baseado na análise das distribuições"""
        logger.info("Iniciando preprocessamento automático...")
        
        # Analisar distribuições se ainda não foi feito
        if self.feature_stats is None or len(self.feature_stats) == 0:
            self.analisar_distribuicoes(X)
        
        X_processed = X.copy()
        
        # Aplicar imputação primeiro
        for _, row in self.feature_stats.iterrows():
            feature = row['feature']
            
            if feature not in X_processed.columns:
                continue
                
            # Imputação
            if row['missing_pct'] > 0:
                imputer_name = row['recomendacao_imputer'] if usar_recomendacoes else 'median'
                imputer = self.imputers[imputer_name]
                
                X_processed[feature] = imputer.fit_transform(
                    X_processed[[feature]]
                ).flatten()
                
                self.fitted_imputers[feature] = imputer
        
        # Aplicar scaling
        for _, row in self.feature_stats.iterrows():
            feature = row['feature']
            
            if feature not in X_processed.columns:
                continue
                
            scaler_name = row['recomendacao_scaler'] if usar_recomendacoes else 'standard'
            scaler = self.scalers[scaler_name]
            
            X_processed[feature] = scaler.fit_transform(
                X_processed[[feature]]
            ).flatten()
            
            self.fitted_scalers[feature] = scaler
        
        logger.info("Preprocessamento automático concluído.")
        return X_processed
    
    def preprocessar_customizado(self, X: pd.DataFrame, 
                               scaler_global: str = 'standard',
                               imputer_global: str = 'median',
                               features_especiais: Dict[str, Dict] = None) -> pd.DataFrame:
        """Preprocessa com configurações customizadas"""
        logger.info(f"Preprocessando com scaler '{scaler_global}' e imputer '{imputer_global}'...")
        
        X_processed = X.copy()
        features_especiais = features_especiais or {}
        
        # Imputação
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                # Usar configuração especial se disponível
                if col in features_especiais and 'imputer' in features_especiais[col]:
                    imputer_name = features_especiais[col]['imputer']
                else:
                    imputer_name = imputer_global
                
                imputer = self.imputers[imputer_name]
                X_processed[col] = imputer.fit_transform(X_processed[[col]]).flatten()
                self.fitted_imputers[col] = imputer
        
        # Scaling
        for col in X_processed.columns:
            # Usar configuração especial se disponível
            if col in features_especiais and 'scaler' in features_especiais[col]:
                scaler_name = features_especiais[col]['scaler']
            else:
                scaler_name = scaler_global
            
            scaler = self.scalers[scaler_name]
            X_processed[col] = scaler.fit_transform(X_processed[[col]]).flatten()
            self.fitted_scalers[col] = scaler
        
        return X_processed
    
    def criar_pipeline_preprocessamento(self, 
                                      features_numericas: List[str],
                                      scaler_type: str = 'standard',
                                      imputer_type: str = 'median') -> Pipeline:
        """Cria pipeline de preprocessamento usando sklearn"""
        logger.info("Criando pipeline de preprocessamento...")
        
        # Pipeline para features numéricas
        numeric_pipeline = Pipeline([
            ('imputer', self.imputers[imputer_type]),
            ('scaler', self.scalers[scaler_type])
        ])
        
        # ColumnTransformer
        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, features_numericas)
        ])
        
        return Pipeline([('preprocessor', preprocessor)])
    
    def transformar_novos_dados(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma novos dados usando scalers já ajustados"""
        logger.info("Transformando novos dados...")
        
        if not self.fitted_scalers:
            raise ValueError("Nenhum scaler foi ajustado. Execute preprocessamento primeiro.")
        
        X_transformed = X.copy()
        
        # Aplicar imputação
        for col in X_transformed.columns:
            if col in self.fitted_imputers:
                X_transformed[col] = self.fitted_imputers[col].transform(
                    X_transformed[[col]]
                ).flatten()
        
        # Aplicar scaling
        for col in X_transformed.columns:
            if col in self.fitted_scalers:
                X_transformed[col] = self.fitted_scalers[col].transform(
                    X_transformed[[col]]
                ).flatten()
        
        return X_transformed
    
    def detectar_outliers(self, X: pd.DataFrame, 
                         metodo: str = 'iqr',
                         threshold: float = 1.5) -> pd.DataFrame:
        """Detecta outliers nas features"""
        logger.info(f"Detectando outliers usando método '{metodo}'...")
        
        outliers_info = []
        
        for col in X.columns:
            serie = X[col].dropna()
            
            if metodo == 'iqr':
                Q1 = serie.quantile(0.25)
                Q3 = serie.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (serie < lower_bound) | (serie > upper_bound)
                
            elif metodo == 'zscore':
                z_scores = np.abs((serie - serie.mean()) / serie.std())
                outliers = z_scores > threshold
                
            elif metodo == 'percentile':
                lower_bound = serie.quantile(threshold / 100)
                upper_bound = serie.quantile(1 - threshold / 100)
                outliers = (serie < lower_bound) | (serie > upper_bound)
            
            outliers_info.append({
                'feature': col,
                'n_outliers': outliers.sum(),
                'pct_outliers': outliers.mean() * 100,
                'lower_bound': lower_bound if metodo != 'zscore' else None,
                'upper_bound': upper_bound if metodo != 'zscore' else None
            })
        
        return pd.DataFrame(outliers_info)
    
    def tratar_outliers(self, X: pd.DataFrame, 
                       metodo: str = 'clip',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Trata outliers nas features"""
        logger.info(f"Tratando outliers usando método '{metodo}'...")
        
        X_treated = X.copy()
        
        for col in X_treated.columns:
            serie = X_treated[col]
            
            # Detectar outliers usando IQR
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            if metodo == 'clip':
                # Clipar valores
                X_treated[col] = serie.clip(lower_bound, upper_bound)
                
            elif metodo == 'remove':
                # Marcar outliers como NaN (serão tratados pela imputação)
                outliers = (serie < lower_bound) | (serie > upper_bound)
                X_treated.loc[outliers, col] = np.nan
                
            elif metodo == 'transform':
                # Transformação log para reduzir impacto
                if serie.min() > 0:
                    X_treated[col] = np.log1p(serie)
        
        return X_treated
    
    def salvar_preprocessadores(self, caminho: str):
        """Salva os preprocessadores ajustados"""
        logger.info(f"Salvando preprocessadores em {caminho}...")
        
        os.makedirs(caminho, exist_ok=True)
        
        # Salvar scalers
        joblib.dump(self.fitted_scalers, os.path.join(caminho, 'scalers.pkl'))
        
        # Salvar imputers
        joblib.dump(self.fitted_imputers, os.path.join(caminho, 'imputers.pkl'))
        
        # Salvar estatísticas
        if hasattr(self, 'feature_stats') and self.feature_stats is not None:
            self.feature_stats.to_csv(os.path.join(caminho, 'feature_stats.csv'), index=False)
    
    def carregar_preprocessadores(self, caminho: str):
        """Carrega preprocessadores salvos"""
        logger.info(f"Carregando preprocessadores de {caminho}...")
        
        # Carregar scalers
        scalers_path = os.path.join(caminho, 'scalers.pkl')
        if os.path.exists(scalers_path):
            self.fitted_scalers = joblib.load(scalers_path)
        
        # Carregar imputers
        imputers_path = os.path.join(caminho, 'imputers.pkl')
        if os.path.exists(imputers_path):
            self.fitted_imputers = joblib.load(imputers_path)
        
        # Carregar estatísticas
        stats_path = os.path.join(caminho, 'feature_stats.csv')
        if os.path.exists(stats_path):
            self.feature_stats = pd.read_csv(stats_path)
    
    def gerar_relatorio_preprocessamento(self, X_original: pd.DataFrame, 
                                       X_processado: pd.DataFrame) -> Dict:
        """Gera relatório do preprocessamento realizado"""
        logger.info("Gerando relatório de preprocessamento...")
        
        relatorio = {
            'features_processadas': len(X_processado.columns),
            'scalers_utilizados': {},
            'imputers_utilizados': {},
            'mudancas_distribuicao': {},
            'outliers_tratados': {}
        }
        
        # Contar scalers utilizados
        for feature, scaler in self.fitted_scalers.items():
            scaler_name = type(scaler).__name__
            relatorio['scalers_utilizados'][scaler_name] = \
                relatorio['scalers_utilizados'].get(scaler_name, 0) + 1
        
        # Contar imputers utilizados
        for feature, imputer in self.fitted_imputers.items():
            imputer_name = type(imputer).__name__
            relatorio['imputers_utilizados'][imputer_name] = \
                relatorio['imputers_utilizados'].get(imputer_name, 0) + 1
        
        # Analisar mudanças na distribuição
        for col in X_original.columns:
            if col in X_processado.columns:
                orig_stats = {
                    'mean': X_original[col].mean(),
                    'std': X_original[col].std(),
                    'skew': X_original[col].skew()
                }
                
                proc_stats = {
                    'mean': X_processado[col].mean(),
                    'std': X_processado[col].std(),
                    'skew': X_processado[col].skew()
                }
                
                relatorio['mudancas_distribuicao'][col] = {
                    'original': orig_stats,
                    'processado': proc_stats,
                    'mudanca_skew': abs(orig_stats['skew'] - proc_stats['skew'])
                }
        
        return relatorio