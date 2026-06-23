# teste_continuo.py - V5 (corrigido)
import torch
import time
import os
import numpy as np

# CORRIGIDO: import sem prefixo v5
from config import MAX_STEPS, COIN_MOVE_EVERY, CHECKPOINT_DIR
from labirinto import Labirinto, MAZE
from moeda import Moeda
from agente import Agente
from utils import render
from colorama import Fore, Style

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    print(f"Versão: V5 - Visão em Linha Reta")
    
    mundo = Labirinto(MAZE)
    agente = Agente(mundo, device)
    
    # Carrega da pasta v5/models
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_ep1000.pth")
    
    if os.path.exists(checkpoint_path):
        try:
            agente.carregar_checkpoint(checkpoint_path)
            print(f"Modelo carregado: {checkpoint_path}")
        except Exception as e:
            print(f"Erro ao carregar checkpoint: {e}")
            print("Tentando carregar checkpoint_final.pth...")
            checkpoint_final = os.p
            ath.join(CHECKPOINT_DIR, "checkpoint_final.pth")
            if os.path.exists(checkpoint_final):
                agente.carregar_checkpoint(checkpoint_final)
                print(f"Modelo carregado: {checkpoint_final}")
            else:
                print("Nenhum checkpoint encontrado. Treine primeiro com main.py")
                return
    else:
        print("Nenhum checkpoint encontrado. Treine primeiro com main.py")
        return
    
    print("\n" + "="*50)
    print(f"{Style.BRIGHT}🐺 HUNTER vs COIN 🪙")
    print("="*50)
    print(f"\n{Fore.RED}H{Style.RESET_ALL} = Hunter (Caçador) - Vermelho")
    print(f"{Fore.YELLOW}C{Style.RESET_ALL} = Coin (Moeda) - Amarelo")
    print(f"{Fore.GREEN}⬤{Style.RESET_ALL} = Cheiro (verde escuro = perto, verde claro = longe)")
    print("\n" + "-"*50)
    print("Aperte Ctrl+C para parar\n")
    
    total_capturas = 0
    try:
        while True:
            agente.pos = mundo.posicao_vazia()
            agente.repeticoes = 0
            moeda = Moeda(mundo, agente, evitar=agente.pos)
            moeda.atualizar_cheiro()
            
            capturada = False
            for step in range(MAX_STEPS):
                limpar_tela()
                
                estado = agente.obter_estado(moeda.cheiro, moeda)
                acao = agente.escolher_acao(estado, 0.0)
                nova_pos, _ = agente.tentar_mover(acao)
                
                if nova_pos == moeda.pos:
                    capturada = True
                
                if not capturada and (step % COIN_MOVE_EVERY == 0):
                    moeda.mover()
                    moeda.atualizar_cheiro()
                    if moeda.pos == nova_pos:
                        capturada = True
                
                agente.pos = nova_pos
                
                render(
                    mundo, 
                    agente.pos, 
                    moeda.pos, 
                    moeda.cheiro, 
                    f"🪙 Moedas capturadas: {total_capturas} 👃 CHEIRO + 👁️ VISÃO", 
                    step
                )
                
                # Mostra status da visão
                visao = agente._ver_moeda(moeda)
                if visao:
                    dx, dy, dist = visao
                    print(f"\n{Fore.CYAN}👁️  VISÃO ATIVA! Moeda a {dist} passos{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.GREEN}👃 Usando olfato...{Style.RESET_ALL}")
                
                if capturada:
                    total_capturas += 1
                    print(f"\n{Fore.YELLOW}🎯 Moeda capturada! Total: {total_capturas}{Style.RESET_ALL}")
                    time.sleep(0.3)
                    break
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}🛑 Teste interrompido pelo usuário.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 Total de moedas capturadas: {total_capturas}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()