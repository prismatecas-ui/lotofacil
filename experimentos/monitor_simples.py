#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Simples de Monitoramento da Otimização
Monitora o progresso da otimização do Random Forest
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

def verificar_processos_python():
    """Verifica se há processos Python rodando"""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe'],
            capture_output=True, text=True, shell=True
        )
        
        if result.returncode == 0 and 'python.exe' in result.stdout:
            linhas = result.stdout.strip().split('\n')
            processos = [linha for linha in linhas if 'python.exe' in linha.lower()]
            return len(processos)
        return 0
    except Exception:
        return 0

def verificar_arquivos_recentes():
    """Verifica arquivos recentes nas pastas de resultado"""
    pasta_projeto = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pasta_resultados = pasta_projeto / "experimentos" / "resultados"
    pasta_modelos = pasta_projeto / "modelos"
    
    arquivos_recentes = []
    agora = datetime.now()
    
    # Verificar resultados
    if pasta_resultados.exists():
        for arquivo in pasta_resultados.glob('*'):
            if arquivo.is_file():
                try:
                    mod_time = datetime.fromtimestamp(arquivo.stat().st_mtime)
                    if agora - mod_time < timedelta(hours=4):
                        arquivos_recentes.append({
                            'nome': arquivo.name,
                            'pasta': 'resultados',
                            'modificado': mod_time,
                            'tamanho': arquivo.stat().st_size
                        })
                except Exception:
                    continue
    
    # Verificar modelos
    if pasta_modelos.exists():
        for arquivo in pasta_modelos.glob('*.pkl'):
            if arquivo.is_file():
                try:
                    mod_time = datetime.fromtimestamp(arquivo.stat().st_mtime)
                    if agora - mod_time < timedelta(hours=4):
                        arquivos_recentes.append({
                            'nome': arquivo.name,
                            'pasta': 'modelos',
                            'modificado': mod_time,
                            'tamanho': arquivo.stat().st_size
                        })
                except Exception:
                    continue
    
    return sorted(arquivos_recentes, key=lambda x: x['modificado'], reverse=True)

def formatar_tamanho(bytes_size):
    """Formata tamanho em bytes"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def main():
    """Função principal de monitoramento"""
    print("🔍 MONITOR SIMPLES DE OTIMIZAÇÃO")
    print("=" * 50)
    print("Pressione Ctrl+C para sair\n")
    
    inicio = datetime.now()
    
    try:
        while True:
            os.system('cls')  # Limpar tela
            
            print("🔍 MONITOR SIMPLES DE OTIMIZAÇÃO")
            print("=" * 50)
            print(f"⏰ Iniciado: {inicio.strftime('%H:%M:%S')}")
            print(f"🕐 Atual: {datetime.now().strftime('%H:%M:%S')}")
            
            # Tempo decorrido
            tempo_decorrido = datetime.now() - inicio
            print(f"⌛ Tempo decorrido: {str(tempo_decorrido).split('.')[0]}")
            print()
            
            # Verificar processos
            num_processos = verificar_processos_python()
            if num_processos > 0:
                print(f"🚀 PROCESSOS PYTHON ATIVOS: {num_processos}")
                print("   ✅ Otimização provavelmente em andamento")
            else:
                print("❌ NENHUM PROCESSO PYTHON DETECTADO")
                print("   ⚠️  Verifique se a otimização está rodando")
            print()
            
            # Progresso estimado (baseado no tempo)
            tempo_estimado_total = 3 * 3600  # 3 horas em segundos
            progresso_pct = min((tempo_decorrido.total_seconds() / tempo_estimado_total) * 100, 95)
            
            print(f"📊 PROGRESSO ESTIMADO: {progresso_pct:.1f}%")
            
            # Barra de progresso simples
            barra_tamanho = 30
            progresso_barra = int((progresso_pct / 100) * barra_tamanho)
            barra = '█' * progresso_barra + '░' * (barra_tamanho - progresso_barra)
            print(f"   [{barra}] {progresso_pct:.1f}%")
            
            # Tempo restante
            if progresso_pct < 95:
                tempo_restante = timedelta(seconds=tempo_estimado_total - tempo_decorrido.total_seconds())
                print(f"   ⏳ Tempo restante estimado: {str(tempo_restante).split('.')[0]}")
            else:
                print(f"   ⏳ Quase concluído!")
            print()
            
            # Arquivos recentes
            arquivos = verificar_arquivos_recentes()
            if arquivos:
                print("📁 ARQUIVOS RECENTES (últimas 4 horas):")
                for arq in arquivos[:5]:  # Mostrar apenas os 5 mais recentes
                    print(f"   📄 {arq['nome']} ({arq['pasta']})")
                    print(f"      {formatar_tamanho(arq['tamanho'])} - {arq['modificado'].strftime('%H:%M:%S')}")
            else:
                print("📁 Nenhum arquivo recente encontrado")
            print()
            
            # Dicas
            print("💡 INFORMAÇÕES:")
            print("   • Processo pode levar 2-4 horas")
            print("   • GridSearchCV testa muitas combinações")
            print("   • Seja paciente com a otimização")
            print("   • Novos arquivos indicam progresso")
            print()
            
            print("⌨️  Pressione Ctrl+C para sair")
            print("🔄 Atualizando em 30 segundos...")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoramento finalizado!")
        print("Obrigado por usar o monitor!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main()