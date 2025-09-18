import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, RFECV, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import cross_val_score
import logging
from typing import Dict, List, Tuple, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelectorLotofacil:
    """Sistema avançado de seleção de features para Lotofácil"""
    
    def __init__(self):
        self.feature_scores = {}
        self.selected_features = {}
        self.feature_rankings = {}
        
    def calcular_f_score(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calcula F-score para cada feature"""
        logger.info("Calculando F-scores...")
        
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X, y)
        
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'f_score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('f_score', ascending=False)
        
        self.feature_scores['f_score'] = scores_df
        return scores_df
    
    def calcular_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calcula Mutual Information para cada feature"""
        logger.info("Calculando Mutual Information...")
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        self.feature_scores['mutual_info'] = scores_df
        return scores_df
    
    def calcular_importancia_random_forest(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calcula importância usando Random Forest"""
        logger.info("Calculando importância Random Forest...")
        
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
        rf.fit(X, y)
        
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        self.feature_scores['rf_importance'] = scores_df
        return scores_df
    
    def calcular_lasso_coefficients(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calcula coeficientes Lasso para seleção de features"""
        logger.info("Calculando coeficientes Lasso...")
        
        # Normalizar dados para Lasso
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso.fit(X_scaled, y)
        
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'lasso_coef': np.abs(lasso.coef_)
        }).sort_values('lasso_coef', ascending=False)
        
        self.feature_scores['lasso'] = scores_df
        return scores_df
    
    def remover_features_baixa_variancia(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features com baixa variância"""
        logger.info(f"Removendo features com variância < {threshold}...")
        
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Features removidas por baixa variância: {len(X.columns) - len(selected_features)}")
        
        return pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
    
    def selecao_recursiva_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 30) -> pd.DataFrame:
        """Seleção recursiva de features usando Random Forest"""
        logger.info(f"Seleção recursiva para {n_features} features...")
        
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selected_features['rfe'] = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def ranking_combinado(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Cria ranking combinado usando múltiplos critérios"""
        logger.info("Criando ranking combinado de features...")
        
        # Calcular todos os scores
        f_scores = self.calcular_f_score(X, y)
        mi_scores = self.calcular_mutual_information(X, y)
        rf_scores = self.calcular_importancia_random_forest(X, y)
        lasso_scores = self.calcular_lasso_coefficients(X, y)
        
        # Criar rankings (posição normalizada de 0 a 1)
        rankings = pd.DataFrame({'feature': X.columns})
        
        # Adicionar rankings normalizados
        for score_name, scores_df in [
            ('f_score', f_scores),
            ('mutual_info', mi_scores),
            ('rf_importance', rf_scores),
            ('lasso', lasso_scores)
        ]:
            # Criar ranking normalizado (1 = melhor, 0 = pior)
            scores_df['rank'] = scores_df[score_name].rank(ascending=False, pct=True)
            rankings = rankings.merge(
                scores_df[['feature', 'rank']].rename(columns={'rank': f'rank_{score_name}'}),
                on='feature'
            )
        
        # Calcular score combinado (média dos rankings)
        ranking_cols = [col for col in rankings.columns if col.startswith('rank_')]
        rankings['score_combinado'] = rankings[ranking_cols].mean(axis=1)
        
        # Ordenar por score combinado
        rankings = rankings.sort_values('score_combinado', ascending=False)
        
        self.feature_rankings['combinado'] = rankings
        
        return rankings
    
    def selecionar_melhores_features(self, X: pd.DataFrame, y: pd.Series, 
                                   n_features: int = 50, 
                                   metodo: str = 'combinado') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Seleciona as melhores features usando o método especificado"""
        logger.info(f"Selecionando {n_features} melhores features usando método '{metodo}'...")
        
        # Remover features com baixa variância primeiro
        X_filtered = self.remover_features_baixa_variancia(X)
        
        if metodo == 'combinado':
            ranking = self.ranking_combinado(X_filtered, y)
            melhores_features = ranking.head(n_features)['feature'].tolist()
            
        elif metodo == 'f_score':
            scores = self.calcular_f_score(X_filtered, y)
            melhores_features = scores.head(n_features)['feature'].tolist()
            
        elif metodo == 'mutual_info':
            scores = self.calcular_mutual_information(X_filtered, y)
            melhores_features = scores.head(n_features)['feature'].tolist()
            
        elif metodo == 'random_forest':
            scores = self.calcular_importancia_random_forest(X_filtered, y)
            melhores_features = scores.head(n_features)['feature'].tolist()
            
        elif metodo == 'rfe':
            X_selected = self.selecao_recursiva_features(X_filtered, y, n_features)
            return X_selected, pd.DataFrame({'feature': X_selected.columns})
            
        else:
            raise ValueError(f"Método '{metodo}' não reconhecido")
        
        # Retornar dataset com features selecionadas
        X_selected = X_filtered[melhores_features]
        feature_info = pd.DataFrame({'feature': melhores_features})
        
        logger.info(f"Seleção concluída. Features finais: {len(melhores_features)}")
        
        return X_selected, feature_info
    
    def avaliar_features(self, X: pd.DataFrame, y: pd.Series, 
                        n_features_list: List[int] = [10, 20, 30, 50, 100]) -> pd.DataFrame:
        """Avalia performance com diferentes números de features"""
        logger.info("Avaliando performance com diferentes números de features...")
        
        resultados = []
        
        for n_features in n_features_list:
            if n_features > X.shape[1]:
                continue
                
            # Selecionar features
            X_selected, _ = self.selecionar_melhores_features(X, y, n_features, 'combinado')
            
            # Avaliar com Random Forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            scores = cross_val_score(rf, X_selected, y, cv=5, scoring='r2')
            
            resultados.append({
                'n_features': n_features,
                'r2_mean': scores.mean(),
                'r2_std': scores.std()
            })
        
        return pd.DataFrame(resultados)
    
    def gerar_relatorio_features(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Gera relatório completo sobre as features"""
        logger.info("Gerando relatório completo de features...")
        
        relatorio = {
            'total_features': X.shape[1],
            'features_baixa_variancia': 0,
            'correlacoes_altas': 0,
            'scores_individuais': {},
            'ranking_combinado': None,
            'avaliacao_performance': None
        }
        
        # Verificar features com baixa variância
        variancias = X.var()
        relatorio['features_baixa_variancia'] = (variancias < 0.01).sum()
        
        # Verificar correlações altas
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        relatorio['correlacoes_altas'] = (upper_tri > 0.95).sum().sum()
        
        # Calcular scores individuais
        relatorio['scores_individuais']['f_score'] = self.calcular_f_score(X, y)
        relatorio['scores_individuais']['mutual_info'] = self.calcular_mutual_information(X, y)
        relatorio['scores_individuais']['rf_importance'] = self.calcular_importancia_random_forest(X, y)
        
        # Ranking combinado
        relatorio['ranking_combinado'] = self.ranking_combinado(X, y)
        
        # Avaliação de performance
        relatorio['avaliacao_performance'] = self.avaliar_features(X, y)
        
        return relatorio