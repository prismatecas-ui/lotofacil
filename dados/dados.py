import logging
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas import ExcelFile, read_excel


def setup_logger(name, level=logging.INFO):
    """Configura e retorna um logger para o módulo especificado.
    
    Args:
        name (str): Nome do logger
        level: Nível de log (default: INFO)
    
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Configura o handler se ainda não existe
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def carregar_dados(usar_cache=True, guia='Importar_Ciclo'):
    """
    Importa os dados da base de dados. Por padrão usa o cache JSON completo
    com todos os concursos atualizados. Se usar_cache=False, usa o Excel.

    :param usar_cache: Se True, usa o cache JSON com dados completos
    :param guia: Guia do Excel (usado apenas se usar_cache=False)
    :return: DataFrame da base de dados.
    """
    
    if usar_cache:
        # Carrega dados do cache JSON (completo e atualizado)
        try:
            with open('./base/cache_concursos.json', 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Converte para DataFrame
            dados_list = []
            for concurso_num, dados_concurso in cache_data.items():
                dados_list.append(dados_concurso)
            
            dados = pd.DataFrame(dados_list)
            
            # Ordena por concurso
            dados = dados.sort_values('Concurso').reset_index(drop=True)
            
            # Adiciona coluna 'Ganhou' baseada em Ganhadores_Sena
            dados['Ganhou'] = (dados['Ganhadores_Sena'] > 0).astype(int)
            
            print(f"✅ Dados carregados do cache: {len(dados)} concursos (do {dados['Concurso'].min()} ao {dados['Concurso'].max()})")
            return dados
            
        except FileNotFoundError:
            print("⚠️  Cache não encontrado, usando Excel...")
            # Fallback para Excel se cache não existir
    
    # Método original usando Excel (limitado)
    caminho = './base/base_dados.xlsx'
    planilha = ExcelFile(caminho)
    dados = read_excel(planilha, guia)
    print(f"📊 Dados carregados do Excel: {len(dados)} concursos")
    
    return dados


def preparar_dados(base_dados):
    """
    Prepara os dados para gerar o modelo.

    :param base_dados: DataFrame da base de dados.

    :return: os dados de atributos (bolas = x) e os dados
    de classificação (ganhadores = y).
    """

    # Carrega a base de dados
    dados = base_dados.copy()

    # Reajustando a columa de ganhadores para:
    # 1 - concurso com ganhadores | 0 - concurso sem ganhadores
    dados.loc[dados['Ganhou'] > 1, 'Ganhou'] = 1

    # Seleciona todas as linhas mais as colunas das dezenas sorteadas
    # e a coluna de ganhadores
    dados = dados.iloc[:, 2:18]

    # Separando atributos (bolas = x) e classe (ganhadores = y)
    atributos = dados.iloc[:, 0:15].values
    classe = dados.iloc[:, 15].values

    return atributos, classe


def dividir_dados(base_dados, tm_teste=0.1, seed=12):
    """
    Divide a base de dados em treino e teste.
    Default: 90% dos dados para treino e 10% dos dados para teste.

    :param base_dados: DataFrame da base de dados.
    :param tm_teste: define o percentual de dados para teste.
    :param seed: padroniza a randomização dos dados para replicação do modelo.
    :return: os dados de treino, teste e o total de atributos contido na base de dados.
    """

    atributos, classe = preparar_dados(base_dados)
    total_atributos = atributos.shape[1]

    # Dividindo os dados em treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(atributos,
                                                            classe,
                                                            test_size=tm_teste,
                                                            random_state=seed)

    return x_treino, x_teste, y_treino, y_teste, total_atributos

