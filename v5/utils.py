# utils.py - V5 (corrigido - cores disponíveis no colorama)
import os
from colorama import init, Fore, Back, Style

init(autoreset=True)

def render_console(labirinto, pos_cacador, pos_moeda, cheiro, episodio, step):
    height, width = labirinto.tamanho, labirinto.tamanho
    
    # Gradiente de verde (usando cores disponíveis no colorama)
    # Mais escuro = mais perto, mais claro = mais longe
    niveis_cheiro = [
        (Fore.LIGHTGREEN_EX, "·"),   # longe (claro)
        (Fore.LIGHTGREEN_EX, "•"),
        (Fore.GREEN, "◦"),
        (Fore.GREEN, "●"),
        (Fore.GREEN, "◐"),
        (Fore.GREEN, "◑"),           # DARK_GREEN não existe, usamos GREEN
        (Fore.GREEN, "◉"),
        (Fore.GREEN, "⬤")            # perto (GREEN mais escuro não existe)
    ]
    
    def nivel_cheiro(intensidade):
        if intensidade < 0.1: return 0
        if intensidade < 0.2: return 1
        if intensidade < 0.3: return 2
        if intensidade < 0.4: return 3
        if intensidade < 0.5: return 4
        if intensidade < 0.6: return 5
        if intensidade < 0.8: return 6
        return 7
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n{Style.BRIGHT}{episodio} | Passo: {step}{Style.RESET_ALL}\n")
    
    for i in range(height):
        linha = ""
        for j in range(width):
            # Caçador (H) em VERMELHO
            if (i, j) == tuple(pos_cacador):
                linha += Fore.RED + Style.BRIGHT + " H " + Style.RESET_ALL
            # Moeda (C) em AMARELO
            elif (i, j) == tuple(pos_moeda):
                linha += Fore.YELLOW + Style.BRIGHT + " C " + Style.RESET_ALL
            # Paredes em cinza escuro
            elif labirinto.eh_parede(i, j):
                linha += Back.LIGHTBLACK_EX + "   " + Style.RESET_ALL
            # Cheiro (gradiente de verde)
            elif cheiro is not None and cheiro[i, j] != -1:
                intensidade = 1.0 / (cheiro[i, j]**2 + 1)
                idx = nivel_cheiro(intensidade)
                cor, simb = niveis_cheiro[idx]
                linha += cor + f" {simb} " + Style.RESET_ALL
            else:
                linha += "   "
        print(linha)
    
    # Legenda atualizada
    print(f"\n{Style.BRIGHT}Legenda:")
    print(f"  {Fore.RED}H{Style.RESET_ALL} = Hunter (Caçador)")
    print(f"  {Fore.YELLOW}C{Style.RESET_ALL} = Coin (Moeda)")
    print(f"  {Fore.GREEN}⬤{Style.RESET_ALL} = Cheiro (quanto mais escuro, mais perto)")
    print(f"{Style.DIM}  {Fore.LIGHTGREEN_EX}·{Style.RESET_ALL} (longe) → {Fore.GREEN}⬤{Style.RESET_ALL} (perto){Style.RESET_ALL}")

def render(*args, **kwargs):
    render_console(*args, **kwargs)

def plotar_taxa_captura(catches_history, window=100):
    import numpy as np
    if len(catches_history) < window:
        print("Histórico muito curto para calcular a média móvel.")
        return
    catch_rate = np.convolve(catches_history, np.ones(window)/window, mode='valid')
    print("\n--- Taxa de Captura (últimos 100 episódios) ---")
    for i, taxa in enumerate(catch_rate[::10]):
        print(f"Ep {i*10+window:4d}: {taxa*100:5.1f}%")
    if len(catch_rate) > 0:
        print(f"Média final: {catch_rate[-1]*100:.1f}%")