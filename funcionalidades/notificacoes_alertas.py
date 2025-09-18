#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Notificações e Alertas - Lotofácil

Este módulo implementa um sistema completo de notificações e alertas
para o sistema Lotofácil, incluindo diferentes canais de comunicação
e configurações personalizáveis.

Autor: Sistema Lotofácil
Versão: 1.0.0
"""

import sqlite3
import json
import smtplib
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
import schedule
from collections import defaultdict, deque
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import wraps
import os
from jinja2 import Template

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TipoAlerta(Enum):
    """Tipos de alertas disponíveis."""
    NUMERO_QUENTE = "numero_quente"
    NUMERO_FRIO = "numero_frio"
    PADRAO_INCOMUM = "padrao_incomum"
    SEQUENCIA_DETECTADA = "sequencia_detectada"
    FREQUENCIA_ANOMALA = "frequencia_anomala"
    NOVO_CONCURSO = "novo_concurso"
    RESULTADO_DISPONIVEL = "resultado_disponivel"
    SISTEMA_ERRO = "sistema_erro"
    MANUTENCAO = "manutencao"
    PREVISAO_ALTA_CONFIANCA = "previsao_alta_confianca"

class PrioridadeAlerta(Enum):
    """Prioridades dos alertas."""
    BAIXA = "baixa"
    MEDIA = "media"
    ALTA = "alta"
    CRITICA = "critica"

class CanalNotificacao(Enum):
    """Canais de notificação disponíveis."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    ARQUIVO = "arquivo"
    CONSOLE = "console"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    PUSH = "push"

@dataclass
class Alerta:
    """Classe para representar um alerta."""
    id: str
    tipo: TipoAlerta
    prioridade: PrioridadeAlerta
    titulo: str
    mensagem: str
    dados: Dict[str, Any]
    timestamp: datetime
    canais: List[CanalNotificacao]
    destinatarios: List[str]
    processado: bool = False
    tentativas: int = 0
    max_tentativas: int = 3
    
    def to_dict(self) -> Dict:
        """Converte o alerta para dicionário."""
        data = asdict(self)
        data['tipo'] = self.tipo.value
        data['prioridade'] = self.prioridade.value
        data['canais'] = [c.value for c in self.canais]
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Alerta':
        """Cria alerta a partir de dicionário."""
        data['tipo'] = TipoAlerta(data['tipo'])
        data['prioridade'] = PrioridadeAlerta(data['prioridade'])
        data['canais'] = [CanalNotificacao(c) for c in data['canais']]
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ConfiguracaoNotificacao:
    """Configuração para notificações."""
    canal: CanalNotificacao
    ativo: bool
    configuracoes: Dict[str, Any]
    tipos_permitidos: List[TipoAlerta]
    prioridade_minima: PrioridadeAlerta
    horario_silencioso: Optional[Dict[str, str]] = None  # {'inicio': '22:00', 'fim': '08:00'}
    dias_ativos: List[int] = None  # 0=segunda, 6=domingo
    
class GeradorAlertas:
    """
    Gerador de alertas baseado em análise de dados.
    """
    
    def __init__(self, db_path: str = "database/lotofacil.db"):
        self.db_path = db_path
        self.historico_alertas = deque(maxlen=1000)
        self.cache_analises = {}
        
    def conectar_db(self) -> sqlite3.Connection:
        """Estabelece conexão com o banco de dados."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Erro ao conectar com o banco: {e}")
            raise
    
    def analisar_frequencias(self, limite: int = 100) -> List[Alerta]:
        """Analisa frequências e gera alertas."""
        alertas = []
        
        try:
            conn = self.conectar_db()
            cursor = conn.cursor()
            
            # Obter dados recentes
            cursor.execute("""
                SELECT numeros_sorteados FROM concursos 
                ORDER BY numero DESC 
                LIMIT ?
            """, (limite,))
            
            resultados = cursor.fetchall()
            conn.close()
            
            if len(resultados) < 10:
                return alertas
            
            # Processar frequências
            todos_numeros = []
            for resultado in resultados:
                numeros = json.loads(resultado['numeros_sorteados'])
                todos_numeros.extend(numeros)
            
            from collections import Counter
            frequencias = Counter(todos_numeros)
            
            # Calcular estatísticas
            media_freq = len(todos_numeros) / 25
            desvio_limite = media_freq * 0.3  # 30% de desvio
            
            # Detectar números quentes
            for numero, freq in frequencias.items():
                if freq > media_freq + desvio_limite:
                    alerta_id = self._gerar_id_alerta(f"quente_{numero}_{freq}")
                    
                    if not self._alerta_ja_enviado(alerta_id, horas=24):
                        alertas.append(Alerta(
                            id=alerta_id,
                            tipo=TipoAlerta.NUMERO_QUENTE,
                            prioridade=PrioridadeAlerta.MEDIA,
                            titulo=f"Número Quente Detectado: {numero}",
                            mensagem=f"O número {numero} apareceu {freq} vezes nos últimos {limite} concursos (média: {media_freq:.1f})",
                            dados={
                                'numero': numero,
                                'frequencia': freq,
                                'media': media_freq,
                                'desvio': freq - media_freq,
                                'concursos_analisados': limite
                            },
                            timestamp=datetime.now(),
                            canais=[CanalNotificacao.EMAIL, CanalNotificacao.CONSOLE],
                            destinatarios=[]
                        ))
            
            # Detectar números frios
            for numero in range(1, 26):
                freq = frequencias.get(numero, 0)
                if freq < media_freq - desvio_limite:
                    alerta_id = self._gerar_id_alerta(f"frio_{numero}_{freq}")
                    
                    if not self._alerta_ja_enviado(alerta_id, horas=24):
                        alertas.append(Alerta(
                            id=alerta_id,
                            tipo=TipoAlerta.NUMERO_FRIO,
                            prioridade=PrioridadeAlerta.BAIXA,
                            titulo=f"Número Frio Detectado: {numero}",
                            mensagem=f"O número {numero} apareceu apenas {freq} vezes nos últimos {limite} concursos (média: {media_freq:.1f})",
                            dados={
                                'numero': numero,
                                'frequencia': freq,
                                'media': media_freq,
                                'desvio': media_freq - freq,
                                'concursos_analisados': limite
                            },
                            timestamp=datetime.now(),
                            canais=[CanalNotificacao.EMAIL, CanalNotificacao.CONSOLE],
                            destinatarios=[]
                        ))
            
            return alertas
            
        except Exception as e:
            logger.error(f"Erro ao analisar frequências: {e}")
            return []
    
    def analisar_padroes(self, limite: int = 50) -> List[Alerta]:
        """Analisa padrões incomuns nos sorteios."""
        alertas = []
        
        try:
            conn = self.conectar_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT numero, numeros_sorteados FROM concursos 
                 ORDER BY numero DESC 
                LIMIT ?
            """, (limite,))
            
            resultados = cursor.fetchall()
            conn.close()
            
            if len(resultados) < 10:
                return alertas
            
            # Analisar padrões
            padroes_paridade = []
            padroes_soma = []
            sequencias_detectadas = []
            
            for resultado in resultados:
                numeros = sorted(json.loads(resultado['numeros_sorteados']))
                
                # Paridade
                pares = sum(1 for n in numeros if n % 2 == 0)
                padroes_paridade.append(pares)
                
                # Soma
                soma = sum(numeros)
                padroes_soma.append(soma)
                
                # Sequências
                sequencias = self._detectar_sequencias(numeros)
                if sequencias:
                    sequencias_detectadas.append({
                        'concurso': resultado['concurso'],
                        'numeros': numeros,
                        'sequencias': sequencias
                    })
            
            # Verificar anomalias de paridade
            import numpy as np
            media_pares = np.mean(padroes_paridade)
            if padroes_paridade[0] > 10 or padroes_paridade[0] < 4:  # Último concurso
                alerta_id = self._gerar_id_alerta(f"paridade_{padroes_paridade[0]}")
                
                if not self._alerta_ja_enviado(alerta_id, horas=12):
                    alertas.append(Alerta(
                        id=alerta_id,
                        tipo=TipoAlerta.PADRAO_INCOMUM,
                        prioridade=PrioridadeAlerta.ALTA,
                        titulo="Padrão de Paridade Incomum",
                        mensagem=f"Último concurso teve {padroes_paridade[0]} números pares (média histórica: {media_pares:.1f})",
                        dados={
                            'pares_ultimo': padroes_paridade[0],
                            'media_historica': media_pares,
                            'considerado_incomum': True
                        },
                        timestamp=datetime.now(),
                        canais=[CanalNotificacao.EMAIL, CanalNotificacao.WEBHOOK],
                        destinatarios=[]
                    ))
            
            # Verificar sequências
            if sequencias_detectadas:
                ultima_sequencia = sequencias_detectadas[0]
                alerta_id = self._gerar_id_alerta(f"sequencia_{ultima_sequencia['concurso']}")
                
                if not self._alerta_ja_enviado(alerta_id, horas=6):
                    alertas.append(Alerta(
                        id=alerta_id,
                        tipo=TipoAlerta.SEQUENCIA_DETECTADA,
                        prioridade=PrioridadeAlerta.MEDIA,
                        titulo="Sequência Numérica Detectada",
                        mensagem=f"Concurso {ultima_sequencia['concurso']} apresentou sequências: {ultima_sequencia['sequencias']}",
                        dados=ultima_sequencia,
                        timestamp=datetime.now(),
                        canais=[CanalNotificacao.EMAIL, CanalNotificacao.CONSOLE],
                        destinatarios=[]
                    ))
            
            return alertas
            
        except Exception as e:
            logger.error(f"Erro ao analisar padrões: {e}")
            return []
    
    def verificar_novos_concursos(self) -> List[Alerta]:
        """Verifica se há novos concursos disponíveis."""
        alertas = []
        
        try:
            # Verificar último concurso no banco
            conn = self.conectar_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(numero) as ultimo_concurso FROM concursos
            """)
            
            resultado = cursor.fetchone()
            ultimo_concurso_db = resultado['ultimo_concurso'] if resultado else 0
            conn.close()
            
            # Simular verificação de API externa (implementar conforme necessário)
            # Por enquanto, apenas exemplo
            ultimo_concurso_api = ultimo_concurso_db  # Placeholder
            
            if ultimo_concurso_api > ultimo_concurso_db:
                alerta_id = self._gerar_id_alerta(f"novo_concurso_{ultimo_concurso_api}")
                
                alertas.append(Alerta(
                    id=alerta_id,
                    tipo=TipoAlerta.NOVO_CONCURSO,
                    prioridade=PrioridadeAlerta.ALTA,
                    titulo="Novo Concurso Disponível",
                    mensagem=f"Concurso {ultimo_concurso_api} está disponível para atualização",
                    dados={
                        'concurso_novo': ultimo_concurso_api,
                        'ultimo_local': ultimo_concurso_db,
                        'diferenca': ultimo_concurso_api - ultimo_concurso_db
                    },
                    timestamp=datetime.now(),
                    canais=[CanalNotificacao.EMAIL, CanalNotificacao.WEBHOOK, CanalNotificacao.PUSH],
                    destinatarios=[]
                ))
            
            return alertas
            
        except Exception as e:
            logger.error(f"Erro ao verificar novos concursos: {e}")
            return []
    
    def _detectar_sequencias(self, numeros: List[int]) -> List[List[int]]:
        """Detecta sequências numéricas."""
        sequencias = []
        numeros_ordenados = sorted(numeros)
        
        i = 0
        while i < len(numeros_ordenados) - 2:
            sequencia_atual = [numeros_ordenados[i]]
            j = i + 1
            
            while j < len(numeros_ordenados) and numeros_ordenados[j] == numeros_ordenados[j-1] + 1:
                sequencia_atual.append(numeros_ordenados[j])
                j += 1
            
            if len(sequencia_atual) >= 3:  # Sequência de pelo menos 3 números
                sequencias.append(sequencia_atual)
            
            i = j if j > i + 1 else i + 1
        
        return sequencias
    
    def _gerar_id_alerta(self, base: str) -> str:
        """Gera ID único para o alerta."""
        timestamp = datetime.now().strftime("%Y%m%d")
        hash_obj = hashlib.md5(f"{base}_{timestamp}".encode())
        return hash_obj.hexdigest()[:12]
    
    def _alerta_ja_enviado(self, alerta_id: str, horas: int = 24) -> bool:
        """Verifica se alerta já foi enviado recentemente."""
        limite_tempo = datetime.now() - timedelta(hours=horas)
        
        for alerta_historico in self.historico_alertas:
            if (alerta_historico.get('id') == alerta_id and 
                datetime.fromisoformat(alerta_historico.get('timestamp', '')) > limite_tempo):
                return True
        
        return False
    
    def registrar_alerta_enviado(self, alerta: Alerta):
        """Registra alerta no histórico."""
        self.historico_alertas.append({
            'id': alerta.id,
            'tipo': alerta.tipo.value,
            'timestamp': alerta.timestamp.isoformat(),
            'processado': True
        })

class ProcessadorNotificacoes:
    """
    Processador de notificações para diferentes canais.
    """
    
    def __init__(self, configuracoes: Dict[CanalNotificacao, ConfiguracaoNotificacao]):
        self.configuracoes = configuracoes
        self.fila_notificacoes = deque()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.templates = self._carregar_templates()
        
    def _carregar_templates(self) -> Dict[str, Template]:
        """Carrega templates de notificação."""
        templates = {}
        
        # Template de email HTML
        template_email_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ titulo }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { background-color: #007bff; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .priority-alta { border-left: 5px solid #dc3545; }
        .priority-media { border-left: 5px solid #ffc107; }
        .priority-baixa { border-left: 5px solid #28a745; }
        .priority-critica { border-left: 5px solid #6f42c1; }
        .dados { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .footer { margin-top: 20px; padding-top: 15px; border-top: 1px solid #dee2e6; font-size: 12px; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container priority-{{ prioridade }}">
        <div class="header">
            <h2>{{ titulo }}</h2>
            <p>Prioridade: {{ prioridade.upper() }} | Tipo: {{ tipo }}</p>
        </div>
        
        <div class="content">
            <p>{{ mensagem }}</p>
            
            {% if dados %}
            <div class="dados">
                <h4>Dados Adicionais:</h4>
                <ul>
                {% for chave, valor in dados.items() %}
                    <li><strong>{{ chave }}:</strong> {{ valor }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>Sistema Lotofácil - {{ timestamp.strftime('%d/%m/%Y %H:%M:%S') }}</p>
            <p>ID do Alerta: {{ id }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Template de texto simples
        template_texto = """
=== ALERTA LOTOFÁCIL ===

Título: {{ titulo }}
Prioridade: {{ prioridade.upper() }}
Tipo: {{ tipo }}
Data/Hora: {{ timestamp.strftime('%d/%m/%Y %H:%M:%S') }}

Mensagem:
{{ mensagem }}

{% if dados %}
Dados Adicionais:
{% for chave, valor in dados.items() %}
- {{ chave }}: {{ valor }}
{% endfor %}
{% endif %}

ID do Alerta: {{ id }}

=== FIM DO ALERTA ===
        """
        
        templates['email_html'] = Template(template_email_html)
        templates['texto'] = Template(template_texto)
        
        return templates
    
    def processar_alerta(self, alerta: Alerta) -> Dict[CanalNotificacao, bool]:
        """Processa um alerta em todos os canais configurados."""
        resultados = {}
        
        for canal in alerta.canais:
            if canal in self.configuracoes and self.configuracoes[canal].ativo:
                config = self.configuracoes[canal]
                
                # Verificar se o tipo de alerta é permitido
                if alerta.tipo not in config.tipos_permitidos:
                    resultados[canal] = False
                    continue
                
                # Verificar prioridade mínima
                prioridades = [PrioridadeAlerta.BAIXA, PrioridadeAlerta.MEDIA, PrioridadeAlerta.ALTA, PrioridadeAlerta.CRITICA]
                if prioridades.index(alerta.prioridade) < prioridades.index(config.prioridade_minima):
                    resultados[canal] = False
                    continue
                
                # Verificar horário silencioso
                if not self._verificar_horario_permitido(config):
                    resultados[canal] = False
                    continue
                
                # Processar notificação
                try:
                    sucesso = self._enviar_notificacao(canal, alerta, config)
                    resultados[canal] = sucesso
                except Exception as e:
                    logger.error(f"Erro ao enviar notificação via {canal.value}: {e}")
                    resultados[canal] = False
            else:
                resultados[canal] = False
        
        return resultados
    
    def _verificar_horario_permitido(self, config: ConfiguracaoNotificacao) -> bool:
        """Verifica se está dentro do horário permitido."""
        agora = datetime.now()
        
        # Verificar dias da semana
        if config.dias_ativos and agora.weekday() not in config.dias_ativos:
            return False
        
        # Verificar horário silencioso
        if config.horario_silencioso:
            inicio = datetime.strptime(config.horario_silencioso['inicio'], '%H:%M').time()
            fim = datetime.strptime(config.horario_silencioso['fim'], '%H:%M').time()
            hora_atual = agora.time()
            
            if inicio <= fim:  # Mesmo dia
                if inicio <= hora_atual <= fim:
                    return False
            else:  # Atravessa meia-noite
                if hora_atual >= inicio or hora_atual <= fim:
                    return False
        
        return True
    
    def _enviar_notificacao(self, canal: CanalNotificacao, alerta: Alerta, config: ConfiguracaoNotificacao) -> bool:
        """Envia notificação para um canal específico."""
        if canal == CanalNotificacao.EMAIL:
            return self._enviar_email(alerta, config)
        elif canal == CanalNotificacao.WEBHOOK:
            return self._enviar_webhook(alerta, config)
        elif canal == CanalNotificacao.ARQUIVO:
            return self._salvar_arquivo(alerta, config)
        elif canal == CanalNotificacao.CONSOLE:
            return self._exibir_console(alerta, config)
        elif canal == CanalNotificacao.TELEGRAM:
            return self._enviar_telegram(alerta, config)
        else:
            logger.warning(f"Canal {canal.value} não implementado")
            return False
    
    def _enviar_email(self, alerta: Alerta, config: ConfiguracaoNotificacao) -> bool:
        """Envia notificação por email."""
        try:
            smtp_config = config.configuracoes
            
            # Criar mensagem
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alerta.prioridade.value.upper()}] {alerta.titulo}"
            msg['From'] = smtp_config['from']
            msg['To'] = ', '.join(alerta.destinatarios or smtp_config.get('destinatarios_padrao', []))
            
            # Texto simples
            texto = self.templates['texto'].render(
                titulo=alerta.titulo,
                prioridade=alerta.prioridade.value,
                tipo=alerta.tipo.value,
                mensagem=alerta.mensagem,
                dados=alerta.dados,
                timestamp=alerta.timestamp,
                id=alerta.id
            )
            
            # HTML
            html = self.templates['email_html'].render(
                titulo=alerta.titulo,
                prioridade=alerta.prioridade.value,
                tipo=alerta.tipo.value,
                mensagem=alerta.mensagem,
                dados=alerta.dados,
                timestamp=alerta.timestamp,
                id=alerta.id
            )
            
            msg.attach(MIMEText(texto, 'plain', 'utf-8'))
            msg.attach(MIMEText(html, 'html', 'utf-8'))
            
            # Enviar
            with smtplib.SMTP(smtp_config['servidor'], smtp_config['porta']) as server:
                if smtp_config.get('tls', True):
                    server.starttls()
                if smtp_config.get('usuario') and smtp_config.get('senha'):
                    server.login(smtp_config['usuario'], smtp_config['senha'])
                
                server.send_message(msg)
            
            logger.info(f"Email enviado com sucesso para {msg['To']}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar email: {e}")
            return False
    
    def _enviar_webhook(self, alerta: Alerta, config: ConfiguracaoNotificacao) -> bool:
        """Envia notificação via webhook."""
        try:
            webhook_config = config.configuracoes
            
            payload = {
                'alerta': alerta.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'sistema': 'lotofacil'
            }
            
            headers = webhook_config.get('headers', {'Content-Type': 'application/json'})
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=webhook_config.get('timeout', 30)
            )
            
            response.raise_for_status()
            logger.info(f"Webhook enviado com sucesso: {response.status_code}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar webhook: {e}")
            return False
    
    def _salvar_arquivo(self, alerta: Alerta, config: ConfiguracaoNotificacao) -> bool:
        """Salva notificação em arquivo."""
        try:
            arquivo_config = config.configuracoes
            caminho = Path(arquivo_config['diretorio'])
            caminho.mkdir(parents=True, exist_ok=True)
            
            # Nome do arquivo com timestamp
            nome_arquivo = f"alerta_{alerta.timestamp.strftime('%Y%m%d_%H%M%S')}_{alerta.id}.txt"
            arquivo_completo = caminho / nome_arquivo
            
            # Conteúdo
            conteudo = self.templates['texto'].render(
                titulo=alerta.titulo,
                prioridade=alerta.prioridade.value,
                tipo=alerta.tipo.value,
                mensagem=alerta.mensagem,
                dados=alerta.dados,
                timestamp=alerta.timestamp,
                id=alerta.id
            )
            
            # Salvar
            with open(arquivo_completo, 'w', encoding='utf-8') as f:
                f.write(conteudo)
            
            logger.info(f"Alerta salvo em arquivo: {arquivo_completo}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo: {e}")
            return False
    
    def _exibir_console(self, alerta: Alerta, config: ConfiguracaoNotificacao) -> bool:
        """Exibe notificação no console."""
        try:
            # Cores para diferentes prioridades
            cores = {
                PrioridadeAlerta.BAIXA: '\033[92m',      # Verde
                PrioridadeAlerta.MEDIA: '\033[93m',      # Amarelo
                PrioridadeAlerta.ALTA: '\033[91m',       # Vermelho
                PrioridadeAlerta.CRITICA: '\033[95m'     # Magenta
            }
            reset = '\033[0m'
            
            cor = cores.get(alerta.prioridade, '')
            
            print(f"\n{cor}{'='*60}{reset}")
            print(f"{cor}ALERTA LOTOFÁCIL - {alerta.prioridade.value.upper()}{reset}")
            print(f"{cor}{'='*60}{reset}")
            print(f"Título: {alerta.titulo}")
            print(f"Tipo: {alerta.tipo.value}")
            print(f"Data/Hora: {alerta.timestamp.strftime('%d/%m/%Y %H:%M:%S')}")
            print(f"\nMensagem:\n{alerta.mensagem}")
            
            if alerta.dados:
                print("\nDados Adicionais:")
                for chave, valor in alerta.dados.items():
                    print(f"  - {chave}: {valor}")
            
            print(f"\nID: {alerta.id}")
            print(f"{cor}{'='*60}{reset}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao exibir no console: {e}")
            return False
    
    def _enviar_telegram(self, alerta: Alerta, config: ConfiguracaoNotificacao) -> bool:
        """Envia notificação via Telegram."""
        try:
            telegram_config = config.configuracoes
            
            # Formatar mensagem
            mensagem = f"🚨 *{alerta.titulo}*\n\n"
            mensagem += f"📊 *Tipo:* {alerta.tipo.value}\n"
            mensagem += f"⚠️ *Prioridade:* {alerta.prioridade.value.upper()}\n"
            mensagem += f"🕐 *Data/Hora:* {alerta.timestamp.strftime('%d/%m/%Y %H:%M:%S')}\n\n"
            mensagem += f"📝 *Mensagem:*\n{alerta.mensagem}\n\n"
            
            if alerta.dados:
                mensagem += "📋 *Dados Adicionais:*\n"
                for chave, valor in alerta.dados.items():
                    mensagem += f"• {chave}: {valor}\n"
            
            mensagem += f"\n🆔 ID: `{alerta.id}`"
            
            # Enviar via API do Telegram
            url = f"https://api.telegram.org/bot{telegram_config['token']}/sendMessage"
            payload = {
                'chat_id': telegram_config['chat_id'],
                'text': mensagem,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info("Mensagem Telegram enviada com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar Telegram: {e}")
            return False

class SistemaNotificacoes:
    """
    Sistema principal de notificações e alertas.
    """
    
    def __init__(self, db_path: str = "database/lotofacil.db", config_path: str = "config/notificacoes.json"):
        self.db_path = db_path
        self.config_path = config_path
        
        # Componentes
        self.gerador = GeradorAlertas(db_path)
        self.configuracoes = self._carregar_configuracoes()
        self.processador = ProcessadorNotificacoes(self.configuracoes)
        
        # Controle de execução
        self.ativo = False
        self.thread_monitoramento = None
        
        # Estatísticas
        self.estatisticas = {
            'alertas_gerados': 0,
            'alertas_enviados': 0,
            'alertas_falharam': 0,
            'ultimo_processamento': None
        }
    
    def _carregar_configuracoes(self) -> Dict[CanalNotificacao, ConfiguracaoNotificacao]:
        """Carrega configurações de notificação."""
        configuracoes = {}
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                for canal_str, config in config_data.items():
                    canal = CanalNotificacao(canal_str)
                    configuracoes[canal] = ConfiguracaoNotificacao(
                        canal=canal,
                        ativo=config['ativo'],
                        configuracoes=config['configuracoes'],
                        tipos_permitidos=[TipoAlerta(t) for t in config['tipos_permitidos']],
                        prioridade_minima=PrioridadeAlerta(config['prioridade_minima']),
                        horario_silencioso=config.get('horario_silencioso'),
                        dias_ativos=config.get('dias_ativos')
                    )
            else:
                # Configurações padrão
                configuracoes = self._gerar_configuracoes_padrao()
                self._salvar_configuracoes(configuracoes)
                
        except Exception as e:
            logger.error(f"Erro ao carregar configurações: {e}")
            configuracoes = self._gerar_configuracoes_padrao()
        
        return configuracoes
    
    def _gerar_configuracoes_padrao(self) -> Dict[CanalNotificacao, ConfiguracaoNotificacao]:
        """Gera configurações padrão."""
        return {
            CanalNotificacao.CONSOLE: ConfiguracaoNotificacao(
                canal=CanalNotificacao.CONSOLE,
                ativo=True,
                configuracoes={},
                tipos_permitidos=list(TipoAlerta),
                prioridade_minima=PrioridadeAlerta.BAIXA
            ),
            CanalNotificacao.ARQUIVO: ConfiguracaoNotificacao(
                canal=CanalNotificacao.ARQUIVO,
                ativo=True,
                configuracoes={'diretorio': 'logs/alertas'},
                tipos_permitidos=list(TipoAlerta),
                prioridade_minima=PrioridadeAlerta.MEDIA
            )
        }
    
    def _salvar_configuracoes(self, configuracoes: Dict[CanalNotificacao, ConfiguracaoNotificacao]):
        """Salva configurações em arquivo."""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {}
            for canal, config in configuracoes.items():
                config_data[canal.value] = {
                    'ativo': config.ativo,
                    'configuracoes': config.configuracoes,
                    'tipos_permitidos': [t.value for t in config.tipos_permitidos],
                    'prioridade_minima': config.prioridade_minima.value,
                    'horario_silencioso': config.horario_silencioso,
                    'dias_ativos': config.dias_ativos
                }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erro ao salvar configurações: {e}")
    
    def iniciar_monitoramento(self, intervalo: int = 300):
        """Inicia o monitoramento automático."""
        if self.ativo:
            logger.warning("Monitoramento já está ativo")
            return
        
        self.ativo = True
        
        def worker_monitoramento():
            logger.info(f"Monitoramento iniciado (intervalo: {intervalo}s)")
            
            while self.ativo:
                try:
                    self.processar_alertas()
                    time.sleep(intervalo)
                except Exception as e:
                    logger.error(f"Erro no monitoramento: {e}")
                    time.sleep(60)  # Aguardar mais tempo em caso de erro
        
        self.thread_monitoramento = threading.Thread(target=worker_monitoramento, daemon=True)
        self.thread_monitoramento.start()
    
    def parar_monitoramento(self):
        """Para o monitoramento automático."""
        self.ativo = False
        if self.thread_monitoramento:
            self.thread_monitoramento.join(timeout=5)
        logger.info("Monitoramento parado")
    
    def processar_alertas(self) -> Dict[str, int]:
        """Processa todos os alertas pendentes."""
        try:
            # Gerar alertas
            alertas = []
            alertas.extend(self.gerador.analisar_frequencias())
            alertas.extend(self.gerador.analisar_padroes())
            alertas.extend(self.gerador.verificar_novos_concursos())
            
            # Processar alertas
            enviados = 0
            falharam = 0
            
            for alerta in alertas:
                resultados = self.processador.processar_alerta(alerta)
                
                if any(resultados.values()):
                    enviados += 1
                    self.gerador.registrar_alerta_enviado(alerta)
                else:
                    falharam += 1
            
            # Atualizar estatísticas
            self.estatisticas['alertas_gerados'] += len(alertas)
            self.estatisticas['alertas_enviados'] += enviados
            self.estatisticas['alertas_falharam'] += falharam
            self.estatisticas['ultimo_processamento'] = datetime.now().isoformat()
            
            logger.info(f"Processamento concluído: {len(alertas)} gerados, {enviados} enviados, {falharam} falharam")
            
            return {
                'gerados': len(alertas),
                'enviados': enviados,
                'falharam': falharam
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar alertas: {e}")
            return {'gerados': 0, 'enviados': 0, 'falharam': 0}
    
    def enviar_alerta_personalizado(self, tipo: TipoAlerta, titulo: str, mensagem: str, 
                                  dados: Dict = None, prioridade: PrioridadeAlerta = PrioridadeAlerta.MEDIA,
                                  canais: List[CanalNotificacao] = None, destinatarios: List[str] = None) -> bool:
        """Envia um alerta personalizado."""
        try:
            alerta = Alerta(
                id=self.gerador._gerar_id_alerta(f"personalizado_{titulo}"),
                tipo=tipo,
                prioridade=prioridade,
                titulo=titulo,
                mensagem=mensagem,
                dados=dados or {},
                timestamp=datetime.now(),
                canais=canais or [CanalNotificacao.CONSOLE],
                destinatarios=destinatarios or []
            )
            
            resultados = self.processador.processar_alerta(alerta)
            sucesso = any(resultados.values())
            
            if sucesso:
                self.gerador.registrar_alerta_enviado(alerta)
                self.estatisticas['alertas_enviados'] += 1
            else:
                self.estatisticas['alertas_falharam'] += 1
            
            return sucesso
            
        except Exception as e:
            logger.error(f"Erro ao enviar alerta personalizado: {e}")
            return False
    
    def obter_estatisticas(self) -> Dict:
        """Obtém estatísticas do sistema."""
        return self.estatisticas.copy()
    
    def testar_configuracao(self, canal: CanalNotificacao) -> bool:
        """Testa uma configuração de canal."""
        try:
            alerta_teste = Alerta(
                id="teste_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                tipo=TipoAlerta.SISTEMA_ERRO,
                prioridade=PrioridadeAlerta.BAIXA,
                titulo="Teste de Configuração",
                mensagem="Este é um alerta de teste para verificar a configuração do canal.",
                dados={'teste': True, 'canal': canal.value},
                timestamp=datetime.now(),
                canais=[canal],
                destinatarios=[]
            )
            
            resultados = self.processador.processar_alerta(alerta_teste)
            return resultados.get(canal, False)
            
        except Exception as e:
            logger.error(f"Erro ao testar configuração {canal.value}: {e}")
            return False


def exemplo_uso():
    """
    Exemplo de uso do sistema de notificações.
    """
    # Criar sistema
    sistema = SistemaNotificacoes()
    
    print("Sistema de Notificações Lotofácil")
    print("Iniciando monitoramento...")
    
    try:
        # Processar alertas uma vez
        resultado = sistema.processar_alertas()
        print(f"Resultado: {resultado}")
        
        # Enviar alerta personalizado
        sucesso = sistema.enviar_alerta_personalizado(
            tipo=TipoAlerta.SISTEMA_ERRO,
            titulo="Teste do Sistema",
            mensagem="Sistema de notificações funcionando corretamente!",
            dados={'versao': '1.0.0', 'teste': True}
        )
        print(f"Alerta personalizado enviado: {sucesso}")
        
        # Mostrar estatísticas
        stats = sistema.obter_estatisticas()
        print(f"Estatísticas: {stats}")
        
        # Iniciar monitoramento contínuo (descomente para usar)
        # sistema.iniciar_monitoramento(intervalo=60)
        # input("Pressione Enter para parar...")
        # sistema.parar_monitoramento()
        
    except KeyboardInterrupt:
        print("\nSistema finalizado.")
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    exemplo_uso()