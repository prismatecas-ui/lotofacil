"""Modelos SQLAlchemy para o sistema Lotofácil.

Este módulo define os modelos de dados que correspondem às tabelas
criadas no banco SQLite, fornecendo uma interface ORM para interação
com os dados dos sorteios da Lotofácil.

Autor: Sistema de Upgrade Lotofácil
Data: 2025
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Float, Text,
    ForeignKey, Index, CheckConstraint, UniqueConstraint, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates, sessionmaker
from sqlalchemy.sql import func

# Base para todos os modelos
Base = declarative_base()


class Sorteio(Base):
    """Modelo para a tabela de sorteios da Lotofácil.
    
    Armazena informações básicas de cada concurso realizado,
    incluindo data, números sorteados e valores de premiação.
    """
    
    __tablename__ = 'sorteios'
    
    # Chave primária
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Dados básicos do sorteio
    concurso = Column(Integer, nullable=False, unique=True, index=True)
    data_sorteio = Column(DateTime, nullable=False, index=True)
    
    # Números sorteados (armazenados como string separada por vírgulas)
    numeros_sorteados = Column(String(50), nullable=False)
    
    # Valores de premiação
    valor_arrecadado = Column(Float, default=0.0)
    valor_acumulado = Column(Float, default=0.0)
    valor_estimado_proximo = Column(Float, default=0.0)
    
    # Estatísticas do sorteio
    total_ganhadores_15 = Column(Integer, default=0)
    total_ganhadores_14 = Column(Integer, default=0)
    total_ganhadores_13 = Column(Integer, default=0)
    total_ganhadores_12 = Column(Integer, default=0)
    total_ganhadores_11 = Column(Integer, default=0)
    
    valor_premio_15 = Column(Float, default=0.0)
    valor_premio_14 = Column(Float, default=0.0)
    valor_premio_13 = Column(Float, default=0.0)
    valor_premio_12 = Column(Float, default=0.0)
    valor_premio_11 = Column(Float, default=0.0)
    
    # Campos de auditoria
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relacionamentos
    numeros = relationship("NumeroSorteado", back_populates="sorteio", cascade="all, delete-orphan")
    padroes = relationship("PadraoSorteio", back_populates="sorteio", cascade="all, delete-orphan")
    resultados_jogos = relationship("ResultadoJogo", back_populates="sorteio", cascade="all, delete-orphan")
    
    # Índices
    __table_args__ = (
        Index('idx_sorteios_data', 'data_sorteio'),
        Index('idx_sorteios_ano_mes', func.strftime('%Y-%m', 'data_sorteio')),
        CheckConstraint('concurso > 0', name='ck_sorteio_concurso_positivo'),
        CheckConstraint('total_ganhadores_15 >= 0', name='ck_sorteio_ganhadores_positivos'),
    )
    
    @validates('numeros_sorteados')
    def validate_numeros_sorteados(self, key, value):
        """Valida se os números sorteados estão no formato correto."""
        if value:
            numeros = [int(n.strip()) for n in value.split(',')]
            if len(numeros) != 15:
                raise ValueError("Devem ser exatamente 15 números sorteados")
            if not all(1 <= n <= 25 for n in numeros):
                raise ValueError("Números devem estar entre 1 e 25")
            if len(set(numeros)) != 15:
                raise ValueError("Números não podem se repetir")
        return value
    
    def get_numeros_lista(self) -> List[int]:
        """Retorna os números sorteados como lista de inteiros."""
        if self.numeros_sorteados:
            return [int(n.strip()) for n in self.numeros_sorteados.split(',')]
        return []
    
    def __repr__(self):
        return f"<Sorteio(concurso={self.concurso}, data={self.data_sorteio.strftime('%d/%m/%Y')})>"


class NumeroSorteado(Base):
    """Modelo para números individuais de cada sorteio.
    
    Permite análises detalhadas por número e posição,
    facilitando estudos estatísticos e de padrões.
    """
    
    __tablename__ = 'numeros_sorteados'
    
    # Chave primária
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Relacionamento com sorteio
    sorteio_id = Column(Integer, ForeignKey('sorteios.id'), nullable=False)
    concurso = Column(Integer, nullable=False, index=True)
    
    # Dados do número
    numero = Column(Integer, nullable=False, index=True)
    posicao = Column(Integer, nullable=False, index=True)  # 1 a 15
    
    # Campos de auditoria
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relacionamentos
    sorteio = relationship("Sorteio", back_populates="numeros")
    
    # Índices e constraints
    __table_args__ = (
        Index('idx_numeros_concurso', 'concurso'),
        Index('idx_numeros_numero', 'numero'),
        Index('idx_numeros_posicao', 'posicao'),
        UniqueConstraint('sorteio_id', 'posicao', name='uk_numero_posicao_sorteio'),
        CheckConstraint('numero >= 1 AND numero <= 25', name='ck_numero_valido'),
        CheckConstraint('posicao >= 1 AND posicao <= 15', name='ck_posicao_valida'),
    )
    
    def __repr__(self):
        return f"<NumeroSorteado(concurso={self.concurso}, numero={self.numero}, posicao={self.posicao})>"


class EstatisticaNumero(Base):
    """Modelo para estatísticas de cada número da Lotofácil.
    
    Mantém contadores e estatísticas atualizadas para cada
    número de 1 a 25, facilitando análises de frequência.
    """
    
    __tablename__ = 'estatisticas_numeros'
    
    # Chave primária
    numero = Column(Integer, primary_key=True)
    
    # Estatísticas básicas
    total_sorteios = Column(Integer, default=0, nullable=False)
    frequencia_absoluta = Column(Integer, default=0, nullable=False)
    frequencia_relativa = Column(Float, default=0.0, nullable=False)
    
    # Estatísticas de atraso
    ultimo_sorteio = Column(Integer, default=0)  # Último concurso em que saiu
    atraso_atual = Column(Integer, default=0)    # Concursos desde a última saída
    maior_atraso = Column(Integer, default=0)    # Maior atraso já registrado
    menor_atraso = Column(Integer, default=0)    # Menor atraso já registrado
    atraso_medio = Column(Float, default=0.0)    # Atraso médio histórico
    
    # Estatísticas por posição
    freq_posicao_1_5 = Column(Integer, default=0)    # Frequência nas posições 1-5
    freq_posicao_6_10 = Column(Integer, default=0)   # Frequência nas posições 6-10
    freq_posicao_11_15 = Column(Integer, default=0)  # Frequência nas posições 11-15
    
    # Tendências
    tendencia_ultimos_10 = Column(Integer, default=0)  # Saídas nos últimos 10 sorteios
    tendencia_ultimos_50 = Column(Integer, default=0)  # Saídas nos últimos 50 sorteios
    
    # Campos de auditoria
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('numero >= 1 AND numero <= 25', name='ck_estatistica_numero_valido'),
        CheckConstraint('frequencia_absoluta >= 0', name='ck_frequencia_positiva'),
        CheckConstraint('atraso_atual >= 0', name='ck_atraso_positivo'),
    )
    
    def calcular_frequencia_relativa(self, total_concursos: int):
        """Calcula e atualiza a frequência relativa do número."""
        if total_concursos > 0:
            self.frequencia_relativa = (self.frequencia_absoluta / total_concursos) * 100
        else:
            self.frequencia_relativa = 0.0
    
    def __repr__(self):
        return f"<EstatisticaNumero(numero={self.numero}, freq={self.frequencia_absoluta}, atraso={self.atraso_atual})>"


class PadraoSorteio(Base):
    """Modelo para padrões identificados em cada sorteio.
    
    Armazena análises de padrões como soma dos números,
    distribuição par/ímpar, sequências, etc.
    """
    
    __tablename__ = 'padroes_sorteios'
    
    # Chave primária
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Relacionamento com sorteio
    sorteio_id = Column(Integer, ForeignKey('sorteios.id'), nullable=False)
    concurso = Column(Integer, nullable=False, index=True)
    
    # Padrões numéricos
    soma_numeros = Column(Integer, nullable=False, index=True)
    qtd_pares = Column(Integer, nullable=False)
    qtd_impares = Column(Integer, nullable=False)
    
    # Distribuição por dezenas
    qtd_dezena_1 = Column(Integer, default=0)  # 1-10
    qtd_dezena_2 = Column(Integer, default=0)  # 11-20
    qtd_dezena_3 = Column(Integer, default=0)  # 21-25
    
    # Padrões de sequência
    qtd_sequencias = Column(Integer, default=0)     # Números consecutivos
    maior_sequencia = Column(Integer, default=0)    # Maior sequência encontrada
    
    # Padrões de repetição
    numeros_repetidos_ultimo = Column(Integer, default=0)  # Repetidos do último sorteio
    numeros_repetidos_penultimo = Column(Integer, default=0)  # Repetidos do penúltimo
    
    # Análise de bordas
    tem_numero_1 = Column(Boolean, default=False)
    tem_numero_25 = Column(Boolean, default=False)
    
    # Campos de auditoria
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relacionamentos
    sorteio = relationship("Sorteio", back_populates="padroes")
    
    # Índices e constraints
    __table_args__ = (
        Index('idx_padroes_concurso', 'concurso'),
        Index('idx_padroes_soma', 'soma_numeros'),
        UniqueConstraint('sorteio_id', name='uk_padrao_sorteio'),
        CheckConstraint('qtd_pares + qtd_impares = 15', name='ck_total_par_impar'),
        CheckConstraint('soma_numeros >= 120 AND soma_numeros <= 300', name='ck_soma_valida'),
    )
    
    def __repr__(self):
        return f"<PadraoSorteio(concurso={self.concurso}, soma={self.soma_numeros}, pares={self.qtd_pares})>"


class ResultadoJogo(Base):
    """Modelo para resultados de jogos/apostas.
    
    Permite armazenar resultados de jogos simulados ou reais,
    facilitando análises de desempenho de estratégias.
    """
    
    __tablename__ = 'resultados_jogos'
    
    # Chave primária
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Relacionamento com sorteio
    sorteio_id = Column(Integer, ForeignKey('sorteios.id'), nullable=False)
    concurso = Column(Integer, nullable=False, index=True)
    
    # Dados do jogo
    numeros_jogados = Column(String(100), nullable=False)  # Números apostados
    qtd_numeros_jogo = Column(Integer, nullable=False)     # 15, 16, 17, 18, 19 ou 20
    
    # Resultado
    acertos = Column(Integer, nullable=False, index=True)
    ganhou = Column(Boolean, default=False, nullable=False)
    valor_premio = Column(Float, default=0.0)
    
    # Metadados do jogo
    estrategia_usada = Column(String(100))  # Nome da estratégia utilizada
    custo_jogo = Column(Float, default=0.0)
    observacoes = Column(Text)
    
    # Campos de auditoria
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relacionamentos
    sorteio = relationship("Sorteio", back_populates="resultados_jogos")
    
    # Índices e constraints
    __table_args__ = (
        Index('idx_resultados_concurso', 'concurso'),
        Index('idx_resultados_acertos', 'acertos'),
        CheckConstraint('qtd_numeros_jogo >= 15 AND qtd_numeros_jogo <= 20', name='ck_qtd_numeros_valida'),
        CheckConstraint('acertos >= 0 AND acertos <= 15', name='ck_acertos_validos'),
    )
    
    @validates('numeros_jogados')
    def validate_numeros_jogados(self, key, value):
        """Valida se os números jogados estão no formato correto."""
        if value:
            numeros = [int(n.strip()) for n in value.split(',')]
            if not all(1 <= n <= 25 for n in numeros):
                raise ValueError("Números devem estar entre 1 e 25")
            if len(set(numeros)) != len(numeros):
                raise ValueError("Números não podem se repetir")
        return value
    
    def get_numeros_lista(self) -> List[int]:
        """Retorna os números jogados como lista de inteiros."""
        if self.numeros_jogados:
            return [int(n.strip()) for n in self.numeros_jogados.split(',')]
        return []
    
    def __repr__(self):
        return f"<ResultadoJogo(concurso={self.concurso}, acertos={self.acertos}, ganhou={self.ganhou})>"


class Configuracao(Base):
    """Modelo para configurações do sistema.
    
    Armazena configurações globais e parâmetros
    que podem ser alterados durante a execução.
    """
    
    __tablename__ = 'configuracoes'
    
    # Chave primária
    chave = Column(String(100), primary_key=True)
    
    # Dados da configuração
    valor = Column(String(500), nullable=False)
    descricao = Column(String(200))
    tipo_valor = Column(String(20), default='string')  # string, integer, float, boolean
    
    # Campos de auditoria
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    def get_valor_tipado(self):
        """Retorna o valor convertido para o tipo apropriado."""
        if self.tipo_valor == 'integer':
            return int(self.valor)
        elif self.tipo_valor == 'float':
            return float(self.valor)
        elif self.tipo_valor == 'boolean':
            return self.valor.lower() in ('true', '1', 'yes', 'sim')
        else:
            return self.valor
    
    def __repr__(self):
        return f"<Configuracao(chave={self.chave}, valor={self.valor})>"


class LogSistema(Base):
    """Modelo para logs do sistema.
    
    Registra eventos importantes, erros e atividades
    do sistema para auditoria e debugging.
    """
    
    __tablename__ = 'logs_sistema'
    
    # Chave primária
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Dados do log
    data_hora = Column(DateTime, default=func.now(), nullable=False, index=True)
    nivel = Column(String(10), nullable=False, index=True)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    modulo = Column(String(50), nullable=False, index=True)
    mensagem = Column(Text, nullable=False)
    
    # Dados adicionais
    usuario = Column(String(100))
    ip_origem = Column(String(45))  # Suporta IPv6
    dados_extras = Column(Text)     # JSON com dados adicionais
    
    # Índices
    __table_args__ = (
        Index('idx_logs_data', 'data_hora'),
        Index('idx_logs_nivel', 'nivel'),
        Index('idx_logs_modulo', 'modulo'),
    )
    
    def __repr__(self):
        return f"<LogSistema(nivel={self.nivel}, modulo={self.modulo}, data={self.data_hora})>"


# Função utilitária para criar todas as tabelas
def criar_tabelas(engine):
    """Cria todas as tabelas no banco de dados.
    
    Args:
        engine: Engine do SQLAlchemy conectado ao banco
    """
    Base.metadata.create_all(engine)
    print("✅ Todas as tabelas foram criadas com sucesso!")


# Função utilitária para obter informações do esquema
def obter_info_esquema():
    """Retorna informações sobre o esquema definido.
    
    Returns:
        dict: Dicionário com informações das tabelas e relacionamentos
    """
    tabelas = {}
    
    for nome_tabela, tabela in Base.metadata.tables.items():
        colunas = []
        for coluna in tabela.columns:
            info_coluna = {
                'nome': coluna.name,
                'tipo': str(coluna.type),
                'nullable': coluna.nullable,
                'primary_key': coluna.primary_key,
                'foreign_key': bool(coluna.foreign_keys)
            }
            colunas.append(info_coluna)
        
        tabelas[nome_tabela] = {
            'colunas': colunas,
            'total_colunas': len(colunas)
        }
    
    return {
        'total_tabelas': len(tabelas),
        'tabelas': tabelas
    }


class Concurso(Base):
    """Modelo para concursos da Lotofácil.
    
    Representa um concurso específico com seus dados básicos.
    """
    
    __tablename__ = 'concursos'
    
    # Chave primária
    id = Column(Integer, primary_key=True, autoincrement=True)
    numero = Column(Integer, unique=True, nullable=False, index=True)
    
    # Dados do concurso
    data_sorteio = Column(DateTime, nullable=False, index=True)
    numeros_sorteados = Column(String(100), nullable=False)
    
    # Campos de auditoria
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Índices
    __table_args__ = (
        Index('idx_concurso_numero', 'numero'),
        Index('idx_concurso_data', 'data_sorteio'),
    )
    
    def get_numeros_lista(self) -> List[int]:
        """Retorna os números sorteados como lista de inteiros."""
        if self.numeros_sorteados:
            return [int(n.strip()) for n in self.numeros_sorteados.split(',')]
        return []
    
    def __repr__(self):
        return f"<Concurso(numero={self.numero}, data={self.data_sorteio})>"


# Configuração do banco de dados
DATABASE_URL = "sqlite:///database/lotofacil.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


if __name__ == "__main__":
    # Exemplo de uso
    print("=== MODELOS SQLALCHEMY LOTOFÁCIL ===")
    print(f"Total de modelos definidos: {len(Base.__subclasses__())}")
    
    for modelo in Base.__subclasses__():
        print(f"  - {modelo.__name__} ({modelo.__tablename__})")
    
    info = obter_info_esquema()
    print(f"\nTotal de tabelas: {info['total_tabelas']}")
    
    for nome, dados in info['tabelas'].items():
        print(f"  {nome}: {dados['total_colunas']} colunas")