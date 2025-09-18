"""Sistema de atualização automática para dados da Lotofácil.

Este módulo implementa rotinas automáticas para manter os dados
da Lotofácil sempre atualizados com os últimos resultados.
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path
from .caixa_api import CaixaAPI, ConcursoResult
from .realtime_service import RealtimeService, UpdateEvent
from .cache_service import cache_service
from models.database_models import SessionLocal, Concurso
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func

logger = logging.getLogger(__name__)

@dataclass
class UpdateConfig:
    """Configurações para atualização automática."""
    check_interval_minutes: int = 30  # Verificar a cada 30 minutos
    batch_size: int = 50  # Processar em lotes de 50
    max_retries: int = 3
    retry_delay_seconds: int = 60
    enable_notifications: bool = True
    backup_before_update: bool = True
    update_cache_file: bool = True
    cache_file_path: str = "base/cache_concursos.json"
    enable_real_time: bool = True
    weekend_check_interval: int = 60  # Verificar mais frequentemente nos fins de semana

@dataclass
class UpdateStats:
    """Estatísticas de atualização."""
    last_update: Optional[datetime] = None
    total_updates: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    last_concurso_number: Optional[int] = None
    average_update_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class DatabaseUpdater:
    """Gerenciador de atualizações do banco de dados."""
    
    def __init__(self, config: Optional[UpdateConfig] = None):
        self.config = config or UpdateConfig()
        self.api = CaixaAPI()
        self.realtime_service = RealtimeService()
        self.stats = UpdateStats()
        self.is_running = False
        self.update_thread = None
        self.callbacks: List[Callable[[UpdateEvent], None]] = []
        
    def add_callback(self, callback: Callable[[UpdateEvent], None]):
        """Adiciona callback para eventos de atualização."""
        self.callbacks.append(callback)
        logger.info(f"Callback de atualização adicionado: {callback.__name__}")
    
    def _notify_callbacks(self, event: UpdateEvent):
        """Notifica todos os callbacks registrados."""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Erro no callback {callback.__name__}: {e}")
    
    def _get_last_concurso_from_db(self) -> Optional[int]:
        """Obtém o número do último concurso no banco de dados."""
        try:
            with SessionLocal() as session:
                result = session.query(func.max(Concurso.numero)).scalar()
                return result
        except SQLAlchemyError as e:
            logger.error(f"Erro ao consultar último concurso no banco: {e}")
            return None
    
    def _save_concurso_to_db(self, concurso: ConcursoResult) -> bool:
        """Salva um concurso no banco de dados."""
        try:
            with SessionLocal() as session:
                # Verificar se já existe
                existing = session.query(Concurso).filter_by(numero=concurso.numero).first()
                
                if existing:
                    # Atualizar dados existentes
                    existing.data_sorteio = concurso.data_sorteio
                    existing.dezenas_sorteadas = json.dumps(concurso.dezenas_sorteadas)
                    existing.acumulou = concurso.acumulou
                    existing.valor_acumulado = concurso.valor_acumulado
                    existing.ganhadores_15_numeros = concurso.ganhadores_15_numeros
                    existing.valor_premio_15_numeros = concurso.valor_premio_15_numeros
                    existing.ganhadores_14_numeros = concurso.ganhadores_14_numeros
                    existing.valor_premio_14_numeros = concurso.valor_premio_14_numeros
                    existing.ganhadores_13_numeros = concurso.ganhadores_13_numeros
                    existing.valor_premio_13_numeros = concurso.valor_premio_13_numeros
                    existing.ganhadores_12_numeros = concurso.ganhadores_12_numeros
                    existing.valor_premio_12_numeros = concurso.valor_premio_12_numeros
                    existing.ganhadores_11_numeros = concurso.ganhadores_11_numeros
                    existing.valor_premio_11_numeros = concurso.valor_premio_11_numeros
                    existing.estimativa_premio = concurso.estimativa_premio
                    existing.valor_arrecadado = concurso.valor_arrecadado
                    existing.updated_at = datetime.now()
                    
                    logger.info(f"Concurso {concurso.numero} atualizado no banco")
                else:
                    # Criar novo registro
                    novo_concurso = Concurso(
                        numero=concurso.numero,
                        data_sorteio=concurso.data_sorteio,
                        dezenas_sorteadas=json.dumps(concurso.dezenas_sorteadas),
                        acumulou=concurso.acumulou,
                        valor_acumulado=concurso.valor_acumulado,
                        ganhadores_15_numeros=concurso.ganhadores_15_numeros,
                        valor_premio_15_numeros=concurso.valor_premio_15_numeros,
                        ganhadores_14_numeros=concurso.ganhadores_14_numeros,
                        valor_premio_14_numeros=concurso.valor_premio_14_numeros,
                        ganhadores_13_numeros=concurso.ganhadores_13_numeros,
                        valor_premio_13_numeros=concurso.valor_premio_13_numeros,
                        ganhadores_12_numeros=concurso.ganhadores_12_numeros,
                        valor_premio_12_numeros=concurso.valor_premio_12_numeros,
                        ganhadores_11_numeros=concurso.ganhadores_11_numeros,
                        valor_premio_11_numeros=concurso.valor_premio_11_numeros,
                        estimativa_premio=concurso.estimativa_premio,
                        valor_arrecadado=concurso.valor_arrecadado,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    session.add(novo_concurso)
                    logger.info(f"Concurso {concurso.numero} inserido no banco")
                
                session.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Erro ao salvar concurso {concurso.numero} no banco: {e}")
            return False
    
    def _update_cache_file(self, concursos: List[ConcursoResult]):
        """Atualiza o arquivo de cache JSON."""
        if not self.config.update_cache_file:
            return
        
        try:
            cache_path = Path(self.config.cache_file_path)
            cache_path.parent.mkdir(exist_ok=True)
            
            # Carregar cache existente
            cache_data = {}
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Atualizar com novos concursos
            for concurso in concursos:
                cache_data[str(concurso.numero)] = {
                    'numero': concurso.numero,
                    'data_sorteio': concurso.data_sorteio.isoformat() if concurso.data_sorteio else None,
                    'dezenas_sorteadas': concurso.dezenas_sorteadas,
                    'acumulou': concurso.acumulou,
                    'valor_acumulado': concurso.valor_acumulado,
                    'ganhadores_15_numeros': concurso.ganhadores_15_numeros,
                    'valor_premio_15_numeros': concurso.valor_premio_15_numeros,
                    'ganhadores_14_numeros': concurso.ganhadores_14_numeros,
                    'valor_premio_14_numeros': concurso.valor_premio_14_numeros,
                    'ganhadores_13_numeros': concurso.ganhadores_13_numeros,
                    'valor_premio_13_numeros': concurso.valor_premio_13_numeros,
                    'ganhadores_12_numeros': concurso.ganhadores_12_numeros,
                    'valor_premio_12_numeros': concurso.valor_premio_12_numeros,
                    'ganhadores_11_numeros': concurso.ganhadores_11_numeros,
                    'valor_premio_11_numeros': concurso.valor_premio_11_numeros,
                    'estimativa_premio': concurso.estimativa_premio,
                    'valor_arrecadado': concurso.valor_arrecadado,
                    'updated_at': datetime.now().isoformat()
                }
            
            # Salvar cache atualizado
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cache atualizado com {len(concursos)} concursos")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar arquivo de cache: {e}")
    
    def check_for_new_concursos(self) -> List[ConcursoResult]:
        """Verifica por novos concursos disponíveis."""
        new_concursos = []
        
        try:
            # Obter último concurso da API
            ultimo_api = self.api.get_ultimo_concurso()
            if not ultimo_api:
                logger.warning("Não foi possível obter último concurso da API")
                return new_concursos
            
            # Obter último concurso do banco
            ultimo_db = self._get_last_concurso_from_db()
            
            if ultimo_db is None:
                logger.info("Banco vazio, iniciando sincronização completa")
                # Se o banco está vazio, buscar os últimos 100 concursos
                inicio = max(1, ultimo_api.numero - 99)
                numeros = list(range(inicio, ultimo_api.numero + 1))
            elif ultimo_api.numero > ultimo_db:
                logger.info(f"Novos concursos detectados: {ultimo_db + 1} a {ultimo_api.numero}")
                numeros = list(range(ultimo_db + 1, ultimo_api.numero + 1))
            else:
                logger.debug("Nenhum novo concurso detectado")
                return new_concursos
            
            # Buscar concursos em lotes
            for i in range(0, len(numeros), self.config.batch_size):
                batch = numeros[i:i + self.config.batch_size]
                logger.info(f"Processando lote: {batch[0]} a {batch[-1]}")
                
                # Buscar concursos do lote
                batch_concursos = self.realtime_service.get_concursos_parallel(batch)
                
                # Filtrar resultados válidos
                for concurso in batch_concursos:
                    if concurso:
                        new_concursos.append(concurso)
                
                # Pequena pausa entre lotes
                time.sleep(1)
            
            logger.info(f"Encontrados {len(new_concursos)} novos concursos")
            return new_concursos
            
        except Exception as e:
            logger.error(f"Erro ao verificar novos concursos: {e}")
            self.stats.errors.append(f"Erro na verificação: {str(e)}")
            return new_concursos
    
    def update_database(self) -> bool:
        """Executa atualização completa do banco de dados."""
        start_time = time.time()
        
        try:
            logger.info("Iniciando atualização do banco de dados")
            
            # Verificar novos concursos
            new_concursos = self.check_for_new_concursos()
            
            if not new_concursos:
                logger.info("Nenhum novo concurso para atualizar")
                return True
            
            # Salvar concursos no banco
            successful_saves = 0
            for concurso in new_concursos:
                if self._save_concurso_to_db(concurso):
                    successful_saves += 1
                    
                    # Notificar callbacks
                    event = UpdateEvent(
                        concurso=concurso,
                        timestamp=datetime.now(),
                        event_type='new',
                        source='auto_update'
                    )
                    self._notify_callbacks(event)
                else:
                    self.stats.failed_updates += 1
            
            # Atualizar arquivo de cache
            if successful_saves > 0:
                self._update_cache_file(new_concursos[:successful_saves])
            
            # Atualizar estatísticas
            self.stats.last_update = datetime.now()
            self.stats.total_updates += len(new_concursos)
            self.stats.successful_updates += successful_saves
            self.stats.last_concurso_number = max(c.numero for c in new_concursos) if new_concursos else None
            
            update_time = time.time() - start_time
            self.stats.average_update_time = (
                (self.stats.average_update_time + update_time) / 2
                if self.stats.average_update_time > 0
                else update_time
            )
            
            logger.info(
                f"Atualização concluída: {successful_saves}/{len(new_concursos)} concursos salvos "
                f"em {update_time:.2f}s"
            )
            
            return successful_saves == len(new_concursos)
            
        except Exception as e:
            logger.error(f"Erro na atualização do banco: {e}")
            self.stats.errors.append(f"Erro na atualização: {str(e)}")
            self.stats.failed_updates += 1
            return False
    
    def start_auto_update(self):
        """Inicia o processo de atualização automática."""
        if self.is_running:
            logger.warning("Atualização automática já está rodando")
            return
        
        self.is_running = True
        logger.info(f"Iniciando atualização automática (intervalo: {self.config.check_interval_minutes}min)")
        
        def update_loop():
            while self.is_running:
                try:
                    # Determinar intervalo baseado no dia da semana
                    now = datetime.now()
                    is_weekend = now.weekday() >= 5  # Sábado ou Domingo
                    
                    interval_minutes = (
                        self.config.weekend_check_interval if is_weekend
                        else self.config.check_interval_minutes
                    )
                    
                    logger.debug(f"Executando verificação automática ({interval_minutes}min)")
                    self.update_database()
                    
                    # Aguardar próxima verificação
                    time.sleep(interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    logger.info("Atualização automática interrompida pelo usuário")
                    break
                except Exception as e:
                    logger.error(f"Erro no loop de atualização: {e}")
                    self.stats.errors.append(f"Erro no loop: {str(e)}")
                    time.sleep(60)  # Aguardar 1 minuto antes de tentar novamente
        
        # Executar em thread separada
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def stop_auto_update(self):
        """Para o processo de atualização automática."""
        self.is_running = False
        logger.info("Atualização automática parada")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da atualização."""
        return {
            'is_running': self.is_running,
            'config': asdict(self.config),
            'stats': asdict(self.stats),
            'last_update_formatted': (
                self.stats.last_update.strftime('%Y-%m-%d %H:%M:%S')
                if self.stats.last_update else None
            )
        }

class ScheduledUpdater:
    """Atualizador baseado em agendamento com horários específicos."""
    
    def __init__(self, config: Optional[UpdateConfig] = None):
        self.config = config or UpdateConfig()
        self.updater = DatabaseUpdater(config)
        self.is_scheduled = False
    
    def schedule_updates(self):
        """Agenda atualizações em horários específicos."""
        if self.is_scheduled:
            logger.warning("Atualizações já estão agendadas")
            return
        
        # Agendar verificações em horários de sorteio
        # Lotofácil: Segunda, Terça, Quinta, Sexta e Sábado às 20h
        schedule.every().monday.at("20:30").do(self._scheduled_update)
        schedule.every().tuesday.at("20:30").do(self._scheduled_update)
        schedule.every().thursday.at("20:30").do(self._scheduled_update)
        schedule.every().friday.at("20:30").do(self._scheduled_update)
        schedule.every().saturday.at("20:30").do(self._scheduled_update)
        
        # Verificações adicionais durante o dia
        schedule.every().day.at("09:00").do(self._scheduled_update)
        schedule.every().day.at("15:00").do(self._scheduled_update)
        
        self.is_scheduled = True
        logger.info("Atualizações agendadas configuradas")
        
        # Iniciar thread do scheduler
        def scheduler_loop():
            while self.is_scheduled:
                schedule.run_pending()
                time.sleep(60)  # Verificar a cada minuto
        
        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()
    
    def _scheduled_update(self):
        """Executa atualização agendada."""
        logger.info("Executando atualização agendada")
        self.updater.update_database()
    
    def stop_scheduled_updates(self):
        """Para as atualizações agendadas."""
        schedule.clear()
        self.is_scheduled = False
        logger.info("Atualizações agendadas paradas")

# Instâncias globais
auto_updater = DatabaseUpdater()
scheduled_updater = ScheduledUpdater()

# Funções de conveniência
def start_auto_update(config: Optional[UpdateConfig] = None) -> DatabaseUpdater:
    """Inicia atualização automática."""
    if config:
        updater = DatabaseUpdater(config)
    else:
        updater = auto_updater
    
    updater.start_auto_update()
    return updater

def stop_auto_update():
    """Para atualização automática."""
    auto_updater.stop_auto_update()

def start_scheduled_updates(config: Optional[UpdateConfig] = None) -> ScheduledUpdater:
    """Inicia atualizações agendadas."""
    if config:
        scheduler = ScheduledUpdater(config)
    else:
        scheduler = scheduled_updater
    
    scheduler.schedule_updates()
    return scheduler

def stop_scheduled_updates():
    """Para atualizações agendadas."""
    scheduled_updater.stop_scheduled_updates()

def manual_update() -> bool:
    """Executa atualização manual."""
    return auto_updater.update_database()

def get_update_stats() -> Dict[str, Any]:
    """Obtém estatísticas de atualização."""
    return auto_updater.get_stats()