#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerenciar download e backup do modelo de IA
Permite usar o sistema em diferentes PCs sem precisar do arquivo de modelo local
"""

import os
import sys
import pickle
import requests
import hashlib
from pathlib import Path
from typing import Optional

# Configurações
MODELO_URL = "https://github.com/seu-usuario/lotofacil-models/releases/download/v1.0/modelo_final_extremo.pkl"
MODELO_LOCAL = "modelos/modelo_final_extremo_20250919_113536.pkl"
MODELO_BACKUP = "backup/modelo_backup.pkl"
CHECKSUM_ESPERADO = ""  # Hash MD5 do modelo original

def calcular_checksum(arquivo: str) -> str:
    """Calcula o checksum MD5 de um arquivo"""
    hash_md5 = hashlib.md5()
    try:
        with open(arquivo, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return ""

def verificar_modelo_local() -> bool:
    """Verifica se o modelo local existe e está íntegro"""
    if not os.path.exists(MODELO_LOCAL):
        print(f"❌ Modelo não encontrado: {MODELO_LOCAL}")
        return False
    
    if CHECKSUM_ESPERADO:
        checksum_atual = calcular_checksum(MODELO_LOCAL)
        if checksum_atual != CHECKSUM_ESPERADO:
            print(f"⚠️  Checksum do modelo não confere. Esperado: {CHECKSUM_ESPERADO}, Atual: {checksum_atual}")
            return False
    
    print(f"✅ Modelo local encontrado e verificado: {MODELO_LOCAL}")
    return True

def baixar_modelo() -> bool:
    """Baixa o modelo do repositório remoto"""
    print(f"📥 Baixando modelo de: {MODELO_URL}")
    
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(MODELO_LOCAL), exist_ok=True)
        
        # Baixar arquivo
        response = requests.get(MODELO_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODELO_LOCAL, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📥 Progresso: {percent:.1f}%", end="")
        
        print(f"\n✅ Modelo baixado com sucesso: {MODELO_LOCAL}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao baixar modelo: {e}")
        return False

def fazer_backup_modelo() -> bool:
    """Cria backup do modelo atual"""
    if not os.path.exists(MODELO_LOCAL):
        print("❌ Modelo local não encontrado para backup")
        return False
    
    try:
        # Criar diretório de backup
        os.makedirs(os.path.dirname(MODELO_BACKUP), exist_ok=True)
        
        # Copiar arquivo
        import shutil
        shutil.copy2(MODELO_LOCAL, MODELO_BACKUP)
        
        print(f"✅ Backup criado: {MODELO_BACKUP}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar backup: {e}")
        return False

def restaurar_backup() -> bool:
    """Restaura modelo do backup"""
    if not os.path.exists(MODELO_BACKUP):
        print("❌ Backup não encontrado")
        return False
    
    try:
        import shutil
        shutil.copy2(MODELO_BACKUP, MODELO_LOCAL)
        print(f"✅ Modelo restaurado do backup")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao restaurar backup: {e}")
        return False

def configurar_modelo() -> bool:
    """Configura o modelo automaticamente"""
    print("🔧 Configurando modelo de IA...")
    
    # 1. Verificar se modelo local existe
    if verificar_modelo_local():
        return True
    
    # 2. Tentar restaurar do backup
    print("🔄 Tentando restaurar do backup...")
    if restaurar_backup() and verificar_modelo_local():
        return True
    
    # 3. Baixar modelo do repositório
    print("🌐 Baixando modelo do repositório...")
    if baixar_modelo() and verificar_modelo_local():
        return True
    
    print("❌ Não foi possível configurar o modelo")
    return False

def main():
    """Função principal"""
    if len(sys.argv) < 2:
        print("Uso: python gerenciar_modelo.py [verificar|baixar|backup|restaurar|configurar]")
        sys.exit(1)
    
    comando = sys.argv[1].lower()
    
    if comando == "verificar":
        if verificar_modelo_local():
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif comando == "baixar":
        if baixar_modelo():
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif comando == "backup":
        if fazer_backup_modelo():
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif comando == "restaurar":
        if restaurar_backup():
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif comando == "configurar":
        if configurar_modelo():
            sys.exit(0)
        else:
            sys.exit(1)
    
    else:
        print(f"Comando desconhecido: {comando}")
        sys.exit(1)

if __name__ == "__main__":
    main()