"""Script de teste para validar os modelos SQLAlchemy.

Testa a criação, relacionamentos e validações dos modelos
definidos para o sistema Lotofácil.

Autor: Sistema de Upgrade Lotofácil
Data: 2025
"""

import os
import sys
from datetime import datetime, date
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# Adiciona o diretório raiz ao path para importar os modelos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database_models import (
    Base, Sorteio, NumeroSorteado, EstatisticaNumero,
    PadraoSorteio, ResultadoJogo, Configuracao, LogSistema,
    criar_tabelas, obter_info_esquema
)


def test_models_creation():
    """Testa a criação dos modelos e tabelas."""
    print("\n=== TESTE 1: CRIAÇÃO DOS MODELOS ===")
    
    # Cria engine em memória para teste
    engine = create_engine('sqlite:///:memory:', echo=False)
    
    try:
        # Cria todas as tabelas
        criar_tabelas(engine)
        print("✅ Tabelas criadas com sucesso")
        
        # Verifica informações do esquema
        info = obter_info_esquema()
        print(f"✅ Esquema validado: {info['total_tabelas']} tabelas")
        
        # Lista as tabelas criadas
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))
            tabelas = [row[0] for row in result.fetchall()]
            
        print(f"✅ Tabelas no banco: {', '.join(tabelas)}")
        
        return engine, True
        
    except Exception as e:
        print(f"❌ Erro na criação: {str(e)}")
        return None, False


def test_basic_operations(engine):
    """Testa operações básicas CRUD."""
    print("\n=== TESTE 2: OPERAÇÕES BÁSICAS ===")
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Teste 1: Criar configuração
        config = Configuracao(
            chave='teste_config',
            valor='valor_teste',
            descricao='Configuração de teste',
            tipo_valor='string'
        )
        session.add(config)
        session.commit()
        print("✅ Configuração criada")
        
        # Teste 2: Criar estatística de número
        stat = EstatisticaNumero(
            numero=1,
            total_sorteios=100,
            frequencia_absoluta=25,
            frequencia_relativa=25.0
        )
        session.add(stat)
        session.commit()
        print("✅ Estatística criada")
        
        # Teste 3: Criar sorteio
        sorteio = Sorteio(
            concurso=1,
            data_sorteio=datetime(2023, 1, 1),
            numeros_sorteados='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15',
            valor_arrecadado=1000000.0,
            total_ganhadores_15=2
        )
        session.add(sorteio)
        session.commit()
        print("✅ Sorteio criado")
        
        # Teste 4: Criar números sorteados
        numeros = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        for i, num in enumerate(numeros, 1):
            numero_sorteado = NumeroSorteado(
                sorteio_id=sorteio.id,
                concurso=1,
                numero=num,
                posicao=i
            )
            session.add(numero_sorteado)
        
        session.commit()
        print("✅ Números sorteados criados")
        
        # Teste 5: Criar padrão do sorteio
        padrao = PadraoSorteio(
            sorteio_id=sorteio.id,
            concurso=1,
            soma_numeros=sum(numeros),
            qtd_pares=len([n for n in numeros if n % 2 == 0]),
            qtd_impares=len([n for n in numeros if n % 2 == 1]),
            qtd_dezena_1=len([n for n in numeros if 1 <= n <= 10]),
            qtd_dezena_2=len([n for n in numeros if 11 <= n <= 20]),
            qtd_dezena_3=len([n for n in numeros if 21 <= n <= 25])
        )
        session.add(padrao)
        session.commit()
        print("✅ Padrão do sorteio criado")
        
        # Teste 6: Criar resultado de jogo
        resultado = ResultadoJogo(
            sorteio_id=sorteio.id,
            concurso=1,
            numeros_jogados='1,2,3,4,5,6,7,8,9,10,11,12,13,14,16',
            qtd_numeros_jogo=15,
            acertos=14,
            ganhou=True,
            valor_premio=1500.0,
            estrategia_usada='Teste Manual'
        )
        session.add(resultado)
        session.commit()
        print("✅ Resultado de jogo criado")
        
        # Teste 7: Criar log do sistema
        log = LogSistema(
            nivel='INFO',
            modulo='teste',
            mensagem='Teste de criação de log',
            usuario='sistema_teste'
        )
        session.add(log)
        session.commit()
        print("✅ Log do sistema criado")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas operações básicas: {str(e)}")
        session.rollback()
        return False
    finally:
        session.close()


def test_relationships(engine):
    """Testa os relacionamentos entre modelos."""
    print("\n=== TESTE 3: RELACIONAMENTOS ===")
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Busca o sorteio criado anteriormente
        sorteio = session.query(Sorteio).filter_by(concurso=1).first()
        
        if not sorteio:
            print("❌ Sorteio não encontrado")
            return False
        
        # Testa relacionamento sorteio -> números
        numeros = sorteio.numeros
        print(f"✅ Sorteio tem {len(numeros)} números relacionados")
        
        # Testa relacionamento sorteio -> padrões
        padroes = sorteio.padroes
        print(f"✅ Sorteio tem {len(padroes)} padrão(ões) relacionado(s)")
        
        # Testa relacionamento sorteio -> resultados
        resultados = sorteio.resultados_jogos
        print(f"✅ Sorteio tem {len(resultados)} resultado(s) relacionado(s)")
        
        # Testa relacionamento inverso número -> sorteio
        if numeros:
            primeiro_numero = numeros[0]
            print(f"✅ Número {primeiro_numero.numero} pertence ao concurso {primeiro_numero.sorteio.concurso}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos relacionamentos: {str(e)}")
        return False
    finally:
        session.close()


def test_validations(engine):
    """Testa as validações dos modelos."""
    print("\n=== TESTE 4: VALIDAÇÕES ===")
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Teste 1: Validação de números sorteados inválidos
        try:
            sorteio_invalido = Sorteio(
                concurso=999,
                data_sorteio=datetime.now(),
                numeros_sorteados='1,2,3,4,5,6,7,8,9,10,11,12,13,14'  # Apenas 14 números
            )
            session.add(sorteio_invalido)
            session.commit()
            print("❌ Validação falhou: deveria rejeitar 14 números")
            return False
        except ValueError:
            print("✅ Validação funcionou: rejeitou números insuficientes")
            session.rollback()
        
        # Teste 2: Validação de números fora do range
        try:
            sorteio_invalido2 = Sorteio(
                concurso=998,
                data_sorteio=datetime.now(),
                numeros_sorteados='1,2,3,4,5,6,7,8,9,10,11,12,13,14,26'  # Número 26 inválido
            )
            session.add(sorteio_invalido2)
            session.commit()
            print("❌ Validação falhou: deveria rejeitar número 26")
            return False
        except ValueError:
            print("✅ Validação funcionou: rejeitou número fora do range")
            session.rollback()
        
        # Teste 3: Validação de constraint de banco
        try:
            numero_invalido = NumeroSorteado(
                sorteio_id=1,
                concurso=1,
                numero=30,  # Número inválido
                posicao=1
            )
            session.add(numero_invalido)
            session.commit()
            print("❌ Constraint falhou: deveria rejeitar número 30")
            return False
        except IntegrityError:
            print("✅ Constraint funcionou: rejeitou número inválido")
            session.rollback()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro inesperado nas validações: {str(e)}")
        return False
    finally:
        session.close()


def test_queries(engine):
    """Testa consultas complexas."""
    print("\n=== TESTE 5: CONSULTAS COMPLEXAS ===")
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Consulta 1: Sorteios com seus números
        sorteios_com_numeros = session.query(Sorteio).join(NumeroSorteado).all()
        print(f"✅ Encontrados {len(sorteios_com_numeros)} sorteios com números")
        
        # Consulta 2: Números mais frequentes
        from sqlalchemy import func
        numeros_freq = session.query(NumeroSorteado.numero, 
                                   func.count(NumeroSorteado.numero).label('freq')
                                   ).group_by(NumeroSorteado.numero).all()
        print(f"✅ Análise de frequência de {len(numeros_freq)} números únicos")
        
        # Consulta 3: Estatísticas por configuração
        configs = session.query(Configuracao).all()
        print(f"✅ Encontradas {len(configs)} configurações")
        
        # Consulta 4: Logs por nível
        logs_info = session.query(LogSistema).filter_by(nivel='INFO').all()
        print(f"✅ Encontrados {len(logs_info)} logs de nível INFO")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas consultas: {str(e)}")
        return False
    finally:
        session.close()


def test_utility_methods():
    """Testa métodos utilitários dos modelos."""
    print("\n=== TESTE 6: MÉTODOS UTILITÁRIOS ===")
    
    try:
        # Teste 1: Método get_numeros_lista do Sorteio
        sorteio = Sorteio(numeros_sorteados='1,5,10,15,20,25,2,7,12,17,22,3,8,13,18')
        numeros_lista = sorteio.get_numeros_lista()
        print(f"✅ get_numeros_lista retornou {len(numeros_lista)} números")
        
        # Teste 2: Método get_valor_tipado da Configuracao
        config_int = Configuracao(chave='teste_int', valor='42', tipo_valor='integer')
        valor_int = config_int.get_valor_tipado()
        print(f"✅ get_valor_tipado converteu para int: {valor_int} (tipo: {type(valor_int).__name__})")
        
        config_bool = Configuracao(chave='teste_bool', valor='true', tipo_valor='boolean')
        valor_bool = config_bool.get_valor_tipado()
        print(f"✅ get_valor_tipado converteu para bool: {valor_bool} (tipo: {type(valor_bool).__name__})")
        
        # Teste 3: Método calcular_frequencia_relativa
        estatistica = EstatisticaNumero(numero=10, frequencia_absoluta=50)
        estatistica.calcular_frequencia_relativa(200)
        print(f"✅ calcular_frequencia_relativa: {estatistica.frequencia_relativa}%")
        
        # Teste 4: Método get_numeros_lista do ResultadoJogo
        resultado = ResultadoJogo(numeros_jogados='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15')
        numeros_jogo = resultado.get_numeros_lista()
        print(f"✅ ResultadoJogo.get_numeros_lista retornou {len(numeros_jogo)} números")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos métodos utilitários: {str(e)}")
        return False


def main():
    """Executa todos os testes dos modelos SQLAlchemy."""
    print("🧪 INICIANDO TESTES DOS MODELOS SQLALCHEMY")
    print("=" * 50)
    
    resultados = []
    
    # Teste 1: Criação dos modelos
    engine, sucesso1 = test_models_creation()
    resultados.append(('Criação dos Modelos', sucesso1))
    
    if not sucesso1 or not engine:
        print("\n💥 Falha crítica na criação dos modelos. Abortando testes.")
        return
    
    # Teste 2: Operações básicas
    sucesso2 = test_basic_operations(engine)
    resultados.append(('Operações Básicas', sucesso2))
    
    # Teste 3: Relacionamentos
    sucesso3 = test_relationships(engine)
    resultados.append(('Relacionamentos', sucesso3))
    
    # Teste 4: Validações
    sucesso4 = test_validations(engine)
    resultados.append(('Validações', sucesso4))
    
    # Teste 5: Consultas
    sucesso5 = test_queries(engine)
    resultados.append(('Consultas Complexas', sucesso5))
    
    # Teste 6: Métodos utilitários
    sucesso6 = test_utility_methods()
    resultados.append(('Métodos Utilitários', sucesso6))
    
    # Resumo dos resultados
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    sucessos = 0
    for nome, sucesso in resultados:
        status = "✅ PASSOU" if sucesso else "❌ FALHOU"
        print(f"{nome:.<30} {status}")
        if sucesso:
            sucessos += 1
    
    print(f"\nResultado Final: {sucessos}/{len(resultados)} testes passaram")
    
    if sucessos == len(resultados):
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Modelos SQLAlchemy validados com sucesso")
        return True
    else:
        print(f"\n⚠️  {len(resultados) - sucessos} teste(s) falharam")
        print("❌ Revisar implementação dos modelos")
        return False


if __name__ == "__main__":
    sucesso_geral = main()
    
    if sucesso_geral:
        print("\n🚀 Modelos prontos para uso em produção!")
    else:
        print("\n🔧 Correções necessárias antes do uso em produção.")