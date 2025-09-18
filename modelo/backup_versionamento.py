#!/usr/bin/env python3
"""
Sistema de Backup e Versionamento de Modelos - Lotofácil

Este módulo implementa um sistema completo de backup e versionamento
para todos os modelos de machine learning do projeto.

Funcionalidades:
- Backup automático de modelos
- Versionamento semântico
- Compressão de arquivos
- Metadados detalhados
- Restauração de versões
- Limpeza automática de backups antigos
- Comparação entre versões
- Relatórios de versionamento
"""

import os
import sys
import json
import shutil
import zipfile
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
from dataclasses import dataclass, asdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VersaoModelo:
    """
    Classe para representar uma versão de modelo
    """
    id: str
    nome_modelo: str
    versao: str
    tipo_modelo: str
    timestamp: str
    arquivo_modelo: str
    arquivo_metadados: str
    arquivo_backup: str
    hash_modelo: str
    tamanho_bytes: int
    metricas: Dict[str, float]
    parametros: Dict[str, Any]
    dependencias: List[str]
    descricao: str
    autor: str
    tags: List[str]
    status: str  # 'ativo', 'arquivado', 'depreciado'
    versao_anterior: Optional[str] = None
    
class GerenciadorBackupVersionamento:
    """
    Gerenciador principal do sistema de backup e versionamento
    """
    
    def __init__(self, diretorio_base: str = "./backups_modelos"):
        self.diretorio_base = Path(diretorio_base)
        self.diretorio_modelos = self.diretorio_base / "modelos"
        self.diretorio_metadados = self.diretorio_base / "metadados"
        self.diretorio_backups = self.diretorio_base / "backups_comprimidos"
        self.arquivo_db = self.diretorio_base / "versionamento.db"
        
        self._criar_estrutura_diretorios()
        self._inicializar_banco_dados()
        self._lock = threading.Lock()
        
    def _criar_estrutura_diretorios(self):
        """
        Cria estrutura de diretórios necessária
        """
        diretorios = [
            self.diretorio_base,
            self.diretorio_modelos,
            self.diretorio_metadados,
            self.diretorio_backups
        ]
        
        for diretorio in diretorios:
            diretorio.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Estrutura de diretórios criada em: {self.diretorio_base}")
    
    def _inicializar_banco_dados(self):
        """
        Inicializa banco de dados SQLite para metadados
        """
        with sqlite3.connect(str(self.arquivo_db)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS versoes_modelos (
                    id TEXT PRIMARY KEY,
                    nome_modelo TEXT NOT NULL,
                    versao TEXT NOT NULL,
                    tipo_modelo TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    arquivo_modelo TEXT NOT NULL,
                    arquivo_metadados TEXT NOT NULL,
                    arquivo_backup TEXT NOT NULL,
                    hash_modelo TEXT NOT NULL,
                    tamanho_bytes INTEGER NOT NULL,
                    metricas TEXT,
                    parametros TEXT,
                    dependencias TEXT,
                    descricao TEXT,
                    autor TEXT,
                    tags TEXT,
                    status TEXT DEFAULT 'ativo',
                    versao_anterior TEXT,
                    UNIQUE(nome_modelo, versao)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historico_operacoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operacao TEXT NOT NULL,
                    modelo_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    detalhes TEXT,
                    usuario TEXT,
                    sucesso BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.commit()
            
        logger.info("Banco de dados inicializado")
    
    def _calcular_hash_arquivo(self, caminho_arquivo: str) -> str:
        """
        Calcula hash SHA256 de um arquivo
        """
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(caminho_arquivo, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Erro ao calcular hash: {e}")
            return ""
    
    def _gerar_id_versao(self, nome_modelo: str, versao: str) -> str:
        """
        Gera ID único para uma versão
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{nome_modelo}_{versao}_{timestamp}"
    
    def _incrementar_versao(self, versao_atual: str, tipo_incremento: str = "patch") -> str:
        """
        Incrementa versão seguindo versionamento semântico
        """
        try:
            partes = versao_atual.split('.')
            major, minor, patch = int(partes[0]), int(partes[1]), int(partes[2])
            
            if tipo_incremento == "major":
                major += 1
                minor = 0
                patch = 0
            elif tipo_incremento == "minor":
                minor += 1
                patch = 0
            else:  # patch
                patch += 1
            
            return f"{major}.{minor}.{patch}"
        
        except Exception:
            # Se não conseguir parsear, usar timestamp
            return datetime.now().strftime('%Y.%m.%d')
    
    def criar_backup_modelo(self, 
                           caminho_modelo: str,
                           nome_modelo: str,
                           tipo_modelo: str,
                           metricas: Dict[str, float] = None,
                           parametros: Dict[str, Any] = None,
                           descricao: str = "",
                           autor: str = "Sistema",
                           tags: List[str] = None,
                           versao: str = None,
                           tipo_incremento: str = "patch") -> VersaoModelo:
        """
        Cria backup completo de um modelo
        """
        with self._lock:
            try:
                # Verificar se arquivo existe
                if not os.path.exists(caminho_modelo):
                    raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")
                
                # Determinar versão
                if versao is None:
                    versao_anterior = self._obter_ultima_versao(nome_modelo)
                    if versao_anterior:
                        versao = self._incrementar_versao(versao_anterior, tipo_incremento)
                    else:
                        versao = "1.0.0"
                
                # Gerar ID único
                id_versao = self._gerar_id_versao(nome_modelo, versao)
                timestamp = datetime.now().isoformat()
                
                # Calcular hash e tamanho
                hash_modelo = self._calcular_hash_arquivo(caminho_modelo)
                tamanho_bytes = os.path.getsize(caminho_modelo)
                
                # Definir caminhos de destino
                nome_arquivo = f"{id_versao}.{Path(caminho_modelo).suffix}"
                caminho_modelo_backup = self.diretorio_modelos / nome_arquivo
                caminho_metadados = self.diretorio_metadados / f"{id_versao}_metadata.json"
                caminho_backup_comprimido = self.diretorio_backups / f"{id_versao}.zip"
                
                # Copiar modelo
                shutil.copy2(caminho_modelo, caminho_modelo_backup)
                
                # Criar objeto versão
                versao_modelo = VersaoModelo(
                    id=id_versao,
                    nome_modelo=nome_modelo,
                    versao=versao,
                    tipo_modelo=tipo_modelo,
                    timestamp=timestamp,
                    arquivo_modelo=str(caminho_modelo_backup),
                    arquivo_metadados=str(caminho_metadados),
                    arquivo_backup=str(caminho_backup_comprimido),
                    hash_modelo=hash_modelo,
                    tamanho_bytes=tamanho_bytes,
                    metricas=metricas or {},
                    parametros=parametros or {},
                    dependencias=self._obter_dependencias(),
                    descricao=descricao,
                    autor=autor,
                    tags=tags or [],
                    status="ativo",
                    versao_anterior=self._obter_ultima_versao(nome_modelo)
                )
                
                # Salvar metadados
                self._salvar_metadados(versao_modelo)
                
                # Criar backup comprimido
                self._criar_backup_comprimido(versao_modelo)
                
                # Salvar no banco de dados
                self._salvar_versao_banco(versao_modelo)
                
                # Registrar operação
                self._registrar_operacao("backup_criado", id_versao, 
                                       f"Backup criado para {nome_modelo} v{versao}", autor)
                
                logger.info(f"Backup criado: {nome_modelo} v{versao} (ID: {id_versao})")
                return versao_modelo
                
            except Exception as e:
                logger.error(f"Erro ao criar backup: {e}")
                self._registrar_operacao("backup_erro", "", str(e), autor, False)
                raise
    
    def _salvar_metadados(self, versao: VersaoModelo):
        """
        Salva metadados da versão em arquivo JSON
        """
        metadados = asdict(versao)
        metadados['created_at'] = datetime.now().isoformat()
        metadados['python_version'] = sys.version
        metadados['sistema_operacional'] = os.name
        
        with open(versao.arquivo_metadados, 'w', encoding='utf-8') as f:
            json.dump(metadados, f, indent=2, ensure_ascii=False)
    
    def _criar_backup_comprimido(self, versao: VersaoModelo):
        """
        Cria arquivo ZIP com modelo e metadados
        """
        with zipfile.ZipFile(versao.arquivo_backup, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Adicionar modelo
            zipf.write(versao.arquivo_modelo, 
                      f"modelo/{Path(versao.arquivo_modelo).name}")
            
            # Adicionar metadados
            zipf.write(versao.arquivo_metadados, 
                      f"metadados/{Path(versao.arquivo_metadados).name}")
            
            # Adicionar arquivo de informações
            info_backup = {
                'versao_sistema': '1.0',
                'data_backup': datetime.now().isoformat(),
                'modelo_info': {
                    'nome': versao.nome_modelo,
                    'versao': versao.versao,
                    'tipo': versao.tipo_modelo,
                    'hash': versao.hash_modelo
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(info_backup, temp_file, indent=2)
                temp_file.flush()
                zipf.write(temp_file.name, "info_backup.json")
                os.unlink(temp_file.name)
    
    def _salvar_versao_banco(self, versao: VersaoModelo):
        """
        Salva versão no banco de dados
        """
        with sqlite3.connect(str(self.arquivo_db)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO versoes_modelos (
                    id, nome_modelo, versao, tipo_modelo, timestamp,
                    arquivo_modelo, arquivo_metadados, arquivo_backup,
                    hash_modelo, tamanho_bytes, metricas, parametros,
                    dependencias, descricao, autor, tags, status, versao_anterior
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                versao.id, versao.nome_modelo, versao.versao, versao.tipo_modelo,
                versao.timestamp, versao.arquivo_modelo, versao.arquivo_metadados,
                versao.arquivo_backup, versao.hash_modelo, versao.tamanho_bytes,
                json.dumps(versao.metricas), json.dumps(versao.parametros),
                json.dumps(versao.dependencias), versao.descricao, versao.autor,
                json.dumps(versao.tags), versao.status, versao.versao_anterior
            ))
            conn.commit()
    
    def _obter_ultima_versao(self, nome_modelo: str) -> Optional[str]:
        """
        Obtém a última versão de um modelo
        """
        with sqlite3.connect(str(self.arquivo_db)) as conn:
            cursor = conn.execute(
                "SELECT versao FROM versoes_modelos WHERE nome_modelo = ? ORDER BY timestamp DESC LIMIT 1",
                (nome_modelo,)
            )
            resultado = cursor.fetchone()
            return resultado[0] if resultado else None
    
    def _obter_dependencias(self) -> List[str]:
        """
        Obtém lista de dependências do ambiente atual
        """
        try:
            import pkg_resources
            return [str(d) for d in pkg_resources.working_set]
        except Exception:
            return []
    
    def _registrar_operacao(self, operacao: str, modelo_id: str, 
                          detalhes: str, usuario: str, sucesso: bool = True):
        """
        Registra operação no histórico
        """
        with sqlite3.connect(str(self.arquivo_db)) as conn:
            conn.execute("""
                INSERT INTO historico_operacoes 
                (operacao, modelo_id, timestamp, detalhes, usuario, sucesso)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (operacao, modelo_id, datetime.now().isoformat(), 
                  detalhes, usuario, sucesso))
            conn.commit()
    
    def listar_versoes(self, nome_modelo: str = None) -> List[VersaoModelo]:
        """
        Lista todas as versões ou de um modelo específico
        """
        with sqlite3.connect(str(self.arquivo_db)) as conn:
            if nome_modelo:
                cursor = conn.execute(
                    "SELECT * FROM versoes_modelos WHERE nome_modelo = ? ORDER BY timestamp DESC",
                    (nome_modelo,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM versoes_modelos ORDER BY timestamp DESC"
                )
            
            versoes = []
            for row in cursor.fetchall():
                versao = VersaoModelo(
                    id=row[0], nome_modelo=row[1], versao=row[2], tipo_modelo=row[3],
                    timestamp=row[4], arquivo_modelo=row[5], arquivo_metadados=row[6],
                    arquivo_backup=row[7], hash_modelo=row[8], tamanho_bytes=row[9],
                    metricas=json.loads(row[10]) if row[10] else {},
                    parametros=json.loads(row[11]) if row[11] else {},
                    dependencias=json.loads(row[12]) if row[12] else [],
                    descricao=row[13] or "", autor=row[14] or "",
                    tags=json.loads(row[15]) if row[15] else [],
                    status=row[16] or "ativo", versao_anterior=row[17]
                )
                versoes.append(versao)
            
            return versoes
    
    def restaurar_versao(self, id_versao: str, caminho_destino: str) -> bool:
        """
        Restaura uma versão específica de um modelo
        """
        try:
            versao = self._obter_versao_por_id(id_versao)
            if not versao:
                raise ValueError(f"Versão não encontrada: {id_versao}")
            
            # Verificar se arquivo de backup existe
            if not os.path.exists(versao.arquivo_backup):
                raise FileNotFoundError(f"Backup não encontrado: {versao.arquivo_backup}")
            
            # Extrair do ZIP
            with zipfile.ZipFile(versao.arquivo_backup, 'r') as zipf:
                # Encontrar arquivo do modelo no ZIP
                arquivos_modelo = [f for f in zipf.namelist() if f.startswith('modelo/')]
                if not arquivos_modelo:
                    raise ValueError("Arquivo de modelo não encontrado no backup")
                
                # Extrair modelo
                zipf.extract(arquivos_modelo[0], tempfile.gettempdir())
                arquivo_temp = os.path.join(tempfile.gettempdir(), arquivos_modelo[0])
                
                # Copiar para destino
                shutil.copy2(arquivo_temp, caminho_destino)
                os.unlink(arquivo_temp)
            
            # Registrar operação
            self._registrar_operacao("restauracao", id_versao, 
                                   f"Modelo restaurado para {caminho_destino}", "Sistema")
            
            logger.info(f"Versão {id_versao} restaurada para {caminho_destino}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao restaurar versão: {e}")
            self._registrar_operacao("restauracao_erro", id_versao, str(e), "Sistema", False)
            return False
    
    def _obter_versao_por_id(self, id_versao: str) -> Optional[VersaoModelo]:
        """
        Obtém versão específica por ID
        """
        versoes = self.listar_versoes()
        for versao in versoes:
            if versao.id == id_versao:
                return versao
        return None
    
    def comparar_versoes(self, id_versao1: str, id_versao2: str) -> Dict[str, Any]:
        """
        Compara duas versões de modelo
        """
        versao1 = self._obter_versao_por_id(id_versao1)
        versao2 = self._obter_versao_por_id(id_versao2)
        
        if not versao1 or not versao2:
            raise ValueError("Uma ou ambas versões não encontradas")
        
        comparacao = {
            'versao1': {
                'id': versao1.id,
                'versao': versao1.versao,
                'timestamp': versao1.timestamp,
                'metricas': versao1.metricas,
                'tamanho_bytes': versao1.tamanho_bytes,
                'hash': versao1.hash_modelo
            },
            'versao2': {
                'id': versao2.id,
                'versao': versao2.versao,
                'timestamp': versao2.timestamp,
                'metricas': versao2.metricas,
                'tamanho_bytes': versao2.tamanho_bytes,
                'hash': versao2.hash_modelo
            },
            'diferencas': {
                'mesmo_hash': versao1.hash_modelo == versao2.hash_modelo,
                'diferenca_tamanho': versao2.tamanho_bytes - versao1.tamanho_bytes,
                'diferenca_metricas': {}
            }
        }
        
        # Comparar métricas
        metricas1 = versao1.metricas
        metricas2 = versao2.metricas
        
        for metrica in set(list(metricas1.keys()) + list(metricas2.keys())):
            valor1 = metricas1.get(metrica, 0)
            valor2 = metricas2.get(metrica, 0)
            comparacao['diferencas']['diferenca_metricas'][metrica] = valor2 - valor1
        
        return comparacao
    
    def limpar_backups_antigos(self, dias_manter: int = 30, manter_minimo: int = 5) -> int:
        """
        Remove backups antigos mantendo um número mínimo
        """
        try:
            data_limite = datetime.now() - timedelta(days=dias_manter)
            
            with sqlite3.connect(str(self.arquivo_db)) as conn:
                # Obter versões antigas por modelo
                cursor = conn.execute("""
                    SELECT nome_modelo, COUNT(*) as total
                    FROM versoes_modelos 
                    GROUP BY nome_modelo
                """)
                
                modelos = cursor.fetchall()
                total_removidos = 0
                
                for nome_modelo, total_versoes in modelos:
                    if total_versoes <= manter_minimo:
                        continue
                    
                    # Obter versões antigas deste modelo
                    cursor = conn.execute("""
                        SELECT id, timestamp, arquivo_modelo, arquivo_metadados, arquivo_backup
                        FROM versoes_modelos 
                        WHERE nome_modelo = ? AND timestamp < ?
                        ORDER BY timestamp ASC
                    """, (nome_modelo, data_limite.isoformat()))
                    
                    versoes_antigas = cursor.fetchall()
                    
                    # Manter pelo menos manter_minimo versões
                    versoes_para_remover = versoes_antigas[:-manter_minimo] if len(versoes_antigas) > manter_minimo else []
                    
                    for versao_antiga in versoes_para_remover:
                        id_versao, timestamp, arquivo_modelo, arquivo_metadados, arquivo_backup = versao_antiga
                        
                        # Remover arquivos físicos
                        for arquivo in [arquivo_modelo, arquivo_metadados, arquivo_backup]:
                            if os.path.exists(arquivo):
                                os.unlink(arquivo)
                        
                        # Remover do banco
                        conn.execute("DELETE FROM versoes_modelos WHERE id = ?", (id_versao,))
                        
                        total_removidos += 1
                        logger.info(f"Backup removido: {id_versao}")
                
                conn.commit()
                
            self._registrar_operacao("limpeza_backups", "", 
                                   f"{total_removidos} backups removidos", "Sistema")
            
            return total_removidos
            
        except Exception as e:
            logger.error(f"Erro ao limpar backups: {e}")
            return 0
    
    def gerar_relatorio_versionamento(self) -> Dict[str, Any]:
        """
        Gera relatório completo do versionamento
        """
        with sqlite3.connect(str(self.arquivo_db)) as conn:
            # Estatísticas gerais
            cursor = conn.execute("SELECT COUNT(*) FROM versoes_modelos")
            total_versoes = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT nome_modelo) FROM versoes_modelos")
            total_modelos = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT SUM(tamanho_bytes) FROM versoes_modelos")
            tamanho_total = cursor.fetchone()[0] or 0
            
            # Modelos por tipo
            cursor = conn.execute("""
                SELECT tipo_modelo, COUNT(*) 
                FROM versoes_modelos 
                GROUP BY tipo_modelo
            """)
            modelos_por_tipo = dict(cursor.fetchall())
            
            # Versões por modelo
            cursor = conn.execute("""
                SELECT nome_modelo, COUNT(*), MAX(timestamp) as ultima_versao
                FROM versoes_modelos 
                GROUP BY nome_modelo
                ORDER BY ultima_versao DESC
            """)
            versoes_por_modelo = cursor.fetchall()
            
            # Histórico de operações recentes
            cursor = conn.execute("""
                SELECT operacao, COUNT(*) 
                FROM historico_operacoes 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY operacao
            """)
            operacoes_recentes = dict(cursor.fetchall())
            
            # Status dos modelos
            cursor = conn.execute("""
                SELECT status, COUNT(*) 
                FROM versoes_modelos 
                GROUP BY status
            """)
            status_modelos = dict(cursor.fetchall())
            
            relatorio = {
                'timestamp': datetime.now().isoformat(),
                'estatisticas_gerais': {
                    'total_versoes': total_versoes,
                    'total_modelos': total_modelos,
                    'tamanho_total_mb': round(tamanho_total / (1024 * 1024), 2),
                    'tamanho_medio_mb': round((tamanho_total / total_versoes) / (1024 * 1024), 2) if total_versoes > 0 else 0
                },
                'modelos_por_tipo': modelos_por_tipo,
                'versoes_por_modelo': [
                    {
                        'nome': nome,
                        'total_versoes': total,
                        'ultima_versao': ultima
                    } for nome, total, ultima in versoes_por_modelo
                ],
                'operacoes_recentes': operacoes_recentes,
                'status_modelos': status_modelos,
                'espaco_disco': {
                    'diretorio_base': str(self.diretorio_base),
                    'tamanho_total_mb': round(tamanho_total / (1024 * 1024), 2)
                }
            }
            
            return relatorio
    
    def exportar_configuracao(self, caminho_arquivo: str):
        """
        Exporta configuração do sistema de versionamento
        """
        configuracao = {
            'versao_sistema': '1.0',
            'diretorio_base': str(self.diretorio_base),
            'estrutura_diretorios': {
                'modelos': str(self.diretorio_modelos),
                'metadados': str(self.diretorio_metadados),
                'backups': str(self.diretorio_backups)
            },
            'banco_dados': str(self.arquivo_db),
            'configuracao_exportada': datetime.now().isoformat()
        }
        
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            json.dump(configuracao, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuração exportada para: {caminho_arquivo}")


def criar_backup_automatico(diretorio_modelos: str, 
                          gerenciador: GerenciadorBackupVersionamento = None) -> List[VersaoModelo]:
    """
    Função utilitária para criar backups automáticos de todos os modelos
    """
    if gerenciador is None:
        gerenciador = GerenciadorBackupVersionamento()
    
    backups_criados = []
    
    # Mapear extensões para tipos de modelo
    tipos_modelo = {
        '.h5': 'tensorflow',
        '.pkl': 'sklearn',
        '.joblib': 'sklearn',
        '.json': 'config',
        '.pb': 'tensorflow_pb'
    }
    
    try:
        for arquivo in Path(diretorio_modelos).rglob('*'):
            if arquivo.is_file() and arquivo.suffix in tipos_modelo:
                nome_modelo = arquivo.stem
                tipo_modelo = tipos_modelo[arquivo.suffix]
                
                try:
                    versao = gerenciador.criar_backup_modelo(
                        str(arquivo),
                        nome_modelo,
                        tipo_modelo,
                        descricao=f"Backup automático de {nome_modelo}",
                        autor="Sistema Automático",
                        tags=["automatico", "backup_completo"]
                    )
                    backups_criados.append(versao)
                    
                except Exception as e:
                    logger.error(f"Erro ao criar backup de {arquivo}: {e}")
    
    except Exception as e:
        logger.error(f"Erro ao processar diretório {diretorio_modelos}: {e}")
    
    return backups_criados


def main():
    """
    Função principal para demonstração
    """
    # Criar gerenciador
    gerenciador = GerenciadorBackupVersionamento()
    
    # Exemplo de uso
    print("Sistema de Backup e Versionamento - Lotofácil")
    print("=" * 50)
    
    # Gerar relatório
    relatorio = gerenciador.gerar_relatorio_versionamento()
    print(f"Total de versões: {relatorio['estatisticas_gerais']['total_versoes']}")
    print(f"Total de modelos: {relatorio['estatisticas_gerais']['total_modelos']}")
    print(f"Tamanho total: {relatorio['estatisticas_gerais']['tamanho_total_mb']} MB")
    
    # Listar versões
    versoes = gerenciador.listar_versoes()
    if versoes:
        print("\nÚltimas versões:")
        for versao in versoes[:5]:
            print(f"  {versao.nome_modelo} v{versao.versao} - {versao.timestamp}")
    
    # Limpeza automática
    removidos = gerenciador.limpar_backups_antigos(dias_manter=30)
    if removidos > 0:
        print(f"\n{removidos} backups antigos removidos")
    
    print("\nSistema de versionamento inicializado com sucesso!")


if __name__ == "__main__":
    main()