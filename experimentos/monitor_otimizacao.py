#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Monitoramento da Otimização do Random Forest
Permite acompanhar o progresso da otimização em tempo real
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

class MonitorOtimizacao:
    def __init__(self):
        self.pasta_projeto = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.pasta_experimentos = self.pasta_projeto / "experimentos"
        self.pasta_resultados = self.pasta_experimentos / "resultados"
        self.pasta_modelos = self.pasta_projeto / "modelos"
        
        # Scripts de otimização a monitorar
        self.scripts_otimizacao = [
            "modelo_final_otimizado.py",
            "otimizar_modelo_avancado.py", 
            "treinar_modelo_completo.py"
        ]
        
        self.inicio_monitoramento = datetime.now()
        
    def limpar_tela(self):
        """Limpa a tela do terminal"""
        os.system('cls')
        
    def verificar_processos_python(self):
        """Verifica processos Python usando tasklist do Windows"""
        processos_encontrados = []
        
        try:
            # Usar tasklist para encontrar processos Python
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                capture_output=True, text=True, shell=True
            )
            
            if result.returncode == 0 and result.stdout:
                linhas = result.stdout.strip().split('\n')
                if len(linhas) > 1:  # Pular cabeçalho
                    for linha in linhas[1:]:
                        if 'python.exe' in linha.lower():
                            partes = linha.split(',')
                            if len(partes) >= 2:
                                pid = partes[1].strip('"')
                                processos_encontrados.append({
                                    'pid': pid,
                                    'nome': 'python.exe',
                                    'encontrado': True
                                })
        except Exception as e:
            print(f"Erro ao verificar processos: {e}")
            
        return processos_encontrados
    
    def verificar_arquivos_recentes(self):
        """Verifica arquivos criados/modificados recentemente"""
        arquivos_recentes = {
            'resultados': [],
            'modelos': []
        }
        
        # Verificar pasta de resultados
        if self.pasta_resultados.exists():
            for arquivo in self.pasta_resultados.glob('*'):
                if arquivo.is_file():
                    try:
                        mod_time = datetime.fromtimestamp(arquivo.stat().st_mtime)
                        if mod_time > self.inicio_monitoramento - timedelta(hours=3):
                            arquivos_recentes['resultados'].append({
                                'nome': arquivo.name,
                                'tamanho': arquivo.stat().st_size,
                                'modificado': mod_time
                            })
                    except Exception:
                        continue
        
        # Verificar pasta de modelos
        if self.pasta_modelos.exists():
            for arquivo in self.pasta_modelos.glob('*.pkl'):
                if arquivo.is_file():
                    try:
                        mod_time = datetime.fromtimestamp(arquivo.stat().st_mtime)
                        if mod_time > self.inicio_monitoramento - timedelta(hours=3):
                            arquivos_recentes['modelos'].append({
                                'nome': arquivo.name,
                                'tamanho': arquivo.stat().st_size,
                                'modificado': mod_time
                            })
                    except Exception:
                        continue
        
        return arquivos_recentes
    
    def verificar_arquivo_log_recente(self):
        """Verifica se há logs recentes de otimização"""
        arquivos_log = []
        
        # Procurar por arquivos de log na pasta experimentos
        for pattern in ['*.log', '*resultado*.json', '*otimizacao*.txt']:
            for arquivo in self.pasta_experimentos.glob(pattern):
                if arquivo.is_file():
                    try:
                        mod_time = datetime.fromtimestamp(arquivo.stat().st_mtime)
                        if mod_time > self.inicio_monitoramento - timedelta(hours=3):
                            arquivos_log.append({
                                'nome': arquivo.name,
                                'tamanho': arquivo.stat().st_size,
                                'modificado': mod_time,
                                'caminho': str(arquivo)
                            })
                    except Exception:
                        continue
        
        return arquivos_log
    
    def formatar_tamanho(self, bytes_size):
        """Formata tamanho em bytes para formato legível"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    def estimar_progresso_por_tempo(self):
        """Estima progresso baseado no tempo decorrido"""
        tempo_decorrido = datetime.now() - self.inicio_monitoramento
        
        # Estimativa baseada em execuções anteriores (2-4 horas)
        tempo_estimado_total = timedelta(hours=3)
        
        progresso_pct = min((tempo_decorrido.total_seconds() / tempo_estimado_total.total_seconds()) * 100, 95)
        tempo_restante = tempo_estimado_total - tempo_decorrido if tempo_decorrido < tempo_estimado_total else timedelta(0)
        
        return {
            'progresso_pct': progresso_pct,
            'tempo_restante': tempo_restante,
            'tempo_decorrido': tempo_decorrido
        }
    
    def exibir_status(self):
        """Exibe o status atual da otimização"""
        self.limpar_tela()
        
        print("🔍 MONITOR DE OTIMIZAÇÃO DO RANDOM FOREST")
        print("=" * 60)
        print(f"⏰ Monitoramento iniciado: {self.inicio_monitoramento.strftime('%H:%M:%S')}")
        print(f"🕐 Hora atual: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Verificar processos Python
        processos = self.verificar_processos_python()
        
        if processos:
            print(f"🚀 PROCESSOS PYTHON ATIVOS: {len(processos)} encontrado(s)")
            for i, proc in enumerate(processos[:3], 1):
                print(f"   📋 Processo {i}: PID {proc['pid']}")
            print()
        else:
            print("❌ NENHUM PROCESSO PYTHON ATIVO DETECTADO")
            print("   Verifique se o script de otimização está rodando.")
            print()
        
        # Progresso estimado por tempo
        progresso = self.estimar_progresso_por_tempo()
        print(f"📊 PROGRESSO ESTIMADO (baseado no tempo):")
        print(f"   📈 Completado: {progresso['progresso_pct']:.1f}%")
        print(f"   ⏳ Tempo restante estimado: {str(progresso['tempo_restante']).split('.')[0]}")
        print(f"   ⌛ Tempo decorrido: {str(progresso['tempo_decorrido']).split('.')[0]}")
        
        # Barra de progresso
        barra_tamanho = 40
        progresso_barra = int((progresso['progresso_pct'] / 100) * barra_tamanho)
        barra = '█' * progresso_barra + '░' * (barra_tamanho - progresso_barra)
        print(f"   [{barra}] {progresso['progresso_pct']:.1f}%")
        print()
        
        # Arquivos recentes
        arquivos = self.verificar_arquivos_recentes()
        logs = self.verificar_arquivo_log_recente()
        
        if any(arquivos.values()) or logs:
            print("📁 ARQUIVOS RECENTES (últimas 3 horas):")
            
            if arquivos['resultados']:
                print("   📊 Resultados:")
                for arq in sorted(arquivos['resultados'], key=lambda x: x['modificado'], reverse=True)[:3]:
                    print(f"      • {arq['nome']} ({self.formatar_tamanho(arq['tamanho'])}) - {arq['modificado'].strftime('%H:%M:%S')}")
            
            if arquivos['modelos']:
                print("   🤖 Modelos:")
                for arq in sorted(arquivos['modelos'], key=lambda x: x['modificado'], reverse=True)[:3]:
                    print(f"      • {arq['nome']} ({self.formatar_tamanho(arq['tamanho'])}) - {arq['modificado'].strftime('%H:%M:%S')}")
            
            if logs:
                print("   📝 Logs/Resultados:")
                for arq in sorted(logs, key=lambda x: x['modificado'], reverse=True)[:3]:
                    print(f"      • {arq['nome']} ({self.formatar_tamanho(arq['tamanho'])}) - {arq['modificado'].strftime('%H:%M:%S')}")
            print()
        else:
            print("📁 Nenhum arquivo recente encontrado nas últimas 3 horas.")
            print()
        
        # Dicas de monitoramento
        print("💡 DICAS DE MONITORAMENTO:")
        print("   • O processo pode levar 2-4 horas para completar")
        print("   • Verifique se há novos arquivos sendo criados")
        print("   • O GridSearchCV testa muitas combinações de parâmetros")
        print("   • Seja paciente - a otimização é um processo demorado")
        print()
        
        print("⌨️  Pressione Ctrl+C para sair")
        print("🔄 Atualizando a cada 30 segundos...")
    
    def monitorar(self):
        """Loop principal de monitoramento"""
        print("🚀 Iniciando monitoramento da otimização...")
        print("⌨️  Pressione Ctrl+C para sair")
        time.sleep(2)
        
        try:
            while True:
                self.exibir_status()
                time.sleep(30)  # Atualiza a cada 30 segundos
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoramento finalizado pelo usuário.")
            print("Obrigado por usar o monitor de otimização!")
            return
        except Exception as e:
            print(f"\n❌ Erro no monitoramento: {e}")
            return

def main():
    """Função principal"""
    try:
        monitor = MonitorOtimizacao()
        monitor.monitorar()
    except Exception as e:
        print(f"Erro ao iniciar monitoramento: {e}")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main()