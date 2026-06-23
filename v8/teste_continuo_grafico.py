# teste_continuo_grafico.py - V8.3 (Gráfico Otimizado, mapa dinâmico)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from config import MAX_STEPS, COIN_MOVE_EVERY, CHECKPOINT_DIR
from labirinto import Labirinto, MAZE
from moeda import Moeda
from agente import Agente
from colorama import Fore, Style

# Configuração da figura
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title('🐺 Hunter vs Coin 🪙 V8.3')
janela_aberta = True

def on_close(event):
    global janela_aberta
    janela_aberta = False
    print("\n🔴 Janela fechada. Teste continua no terminal...")
fig.canvas.mpl_connect('close_event', on_close)

# Variáveis globais para o gráfico
height, width = 25, 25  # será atualizado dinamicamente
heatmap = None
im = None
hunter_patch = None
coin_patch = None
paredes_patches = []

def init_plot(mundo):
    """Inicializa os objetos gráficos com o tamanho do mapa."""
    global height, width, heatmap, im, hunter_patch, coin_patch, paredes_patches
    height, width = mundo.tamanho, mundo.tamanho
    
    # Remove patches antigos
    for patch in paredes_patches:
        patch.remove()
    paredes_patches.clear()
    if hunter_patch is not None:
        hunter_patch.remove()
    if coin_patch is not None:
        coin_patch.remove()
    
    # Mapa de calor
    heatmap = np.zeros((height, width))
    im = ax.imshow(heatmap, cmap='viridis', vmin=0, vmax=1, origin='upper')
    
    # Paredes
    for i in range(height):
        for j in range(width):
            if mundo.eh_parede(i, j):
                patch = plt.Rectangle((j-0.5, i-0.5), 1, 1, color='#2c3e50', lw=0)
                ax.add_patch(patch)
                paredes_patches.append(patch)
    
    # Hunter e Coin
    hunter_patch = plt.Rectangle((0, 0), 0.8, 0.8, color='red', ec='darkred', lw=2)
    coin_patch = plt.Circle((0, 0), 0.4, color='gold', ec='orange', lw=2)
    ax.add_patch(hunter_patch)
    ax.add_patch(coin_patch)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(height-0.5, -0.5)
    plt.tight_layout()
    fig.canvas.draw()

def render_plot(mundo, pos_hunter, pos_coin, cheiro, total_capturas, step, visao_ativa, distancia_visao):
    """Renderiza o gráfico com atualização incremental."""
    global janela_aberta, heatmap, im, hunter_patch, coin_patch
    if not janela_aberta:
        return
    
    # Atualiza mapa de calor
    if cheiro is not None:
        for i in range(height):
            for j in range(width):
                if cheiro[i, j] != -1:
                    heatmap[i, j] = 1.0 / (cheiro[i, j] + 1)
                else:
                    heatmap[i, j] = 0.0
    else:
        heatmap.fill(0.0)
    im.set_data(heatmap)
    
    # Atualiza posição do Hunter
    hx, hy = pos_hunter
    hunter_patch.set_xy((hy-0.4, hx-0.4))
    
    # Atualiza posição da Coin
    cx, cy = pos_coin
    coin_patch.center = (cy, cx)
    
    # Atualiza título
    titulo = f"🪙 Moedas: {total_capturas} | Passo: {step}"
    if visao_ativa:
        titulo += f" | 👁️ Visão! Dist: {distancia_visao}"
    else:
        titulo += " | 👃 Olfato"
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    
    # Redesenha
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.0001)

def main():
    global janela_aberta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    print(f"Versão: V8.3 - Teste Gráfico Otimizado")

    mundo = Labirinto(MAZE)
    agente = Agente(mundo, device)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
    if os.path.exists(checkpoint_path):
        try:
            agente.carregar_checkpoint(checkpoint_path)
            print(f"Modelo carregado: {checkpoint_path}")
        except:
            final_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final.pth")
            if os.path.exists(final_path):
                agente.carregar_checkpoint(final_path)
                print(f"Modelo carregado: {final_path}")
            else:
                print("Nenhum checkpoint encontrado.")
                return
    else:
        print("Nenhum checkpoint encontrado.")
        return

    # Inicializa o gráfico com o tamanho do mapa
    init_plot(mundo)

    print("\n" + "="*50)
    print(f"{Style.BRIGHT}🐺 HUNTER vs COIN 🪙 (V8.3 - Modo Gráfico)")
    print("="*50)
    print("A janela será aberta. Feche-a ou pressione Ctrl+C para parar.")
    print(f"Mapa: {mundo.tamanho}x{mundo.tamanho}\n")

    total_capturas = 0
    try:
        while True:
            agente.pos = mundo.posicao_vazia()
            agente.historico_posicoes.clear()
            agente.contador_loop = 0
            agente.penalidade_loop_acumulada = 0.0
            agente.prob_aleatoria = 0.0

            moeda = Moeda(mundo, agente, evitar=agente.pos)
            moeda.atualizar_cheiro()

            capturada = False
            for step in range(MAX_STEPS):
                # Renderiza a cada 2 passos (equilíbrio entre velocidade e fluidez)
                if step % 2 == 0 and janela_aberta:
                    estado = agente.obter_estado(moeda.cheiro, moeda)
                    acao = agente.escolher_acao(estado, 0.0)
                    nova_pos, bateu, penalidade_loop = agente.tentar_mover(acao)

                    if nova_pos == moeda.pos:
                        capturada = True
                    if not capturada and (step % COIN_MOVE_EVERY == 0):
                        moeda.mover()
                        moeda.atualizar_cheiro()
                        if moeda.pos == nova_pos:
                            capturada = True

                    agente.pos = nova_pos

                    visao = agente._ver_moeda(moeda)
                    visao_ativa = visao is not None
                    distancia_visao = visao[2] if visao_ativa else None

                    render_plot(mundo, agente.pos, moeda.pos, moeda.cheiro,
                                total_capturas, step, visao_ativa, distancia_visao)

                    if capturada:
                        total_capturas += 1
                        print(f"🎯 Capturada! Total: {total_capturas}")
                        time.sleep(0.15)
                        break
                else:
                    # Executa movimento sem renderizar (mais rápido)
                    estado = agente.obter_estado(moeda.cheiro, moeda)
                    acao = agente.escolher_acao(estado, 0.0)
                    nova_pos, bateu, penalidade_loop = agente.tentar_mover(acao)

                    if nova_pos == moeda.pos:
                        capturada = True
                    if not capturada and (step % COIN_MOVE_EVERY == 0):
                        moeda.mover()
                        moeda.atualizar_cheiro()
                        if moeda.pos == nova_pos:
                            capturada = True

                    agente.pos = nova_pos

                    if capturada:
                        total_capturas += 1
                        print(f"🎯 Capturada! Total: {total_capturas}")
                        time.sleep(0.15)
                        break

            if not janela_aberta:
                # Modo terminal - continua rodando
                pass

    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}🛑 Teste interrompido.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 Total capturadas: {total_capturas}{Style.RESET_ALL}")
    except Exception as e:
        if "closed" in str(e).lower():
            print("\n🔴 Janela fechada. Continuando no terminal...")
            try:
                while True:
                    agente.pos = mundo.posicao_vazia()
                    agente.historico_posicoes.clear()
                    agente.contador_loop = 0
                    agente.penalidade_loop_acumulada = 0.0
                    agente.prob_aleatoria = 0.0

                    moeda = Moeda(mundo, agente, evitar=agente.pos)
                    moeda.atualizar_cheiro()

                    capturada = False
                    for step in range(MAX_STEPS):
                        estado = agente.obter_estado(moeda.cheiro, moeda)
                        acao = agente.escolher_acao(estado, 0.0)
                        nova_pos, bateu, penalidade_loop = agente.tentar_mover(acao)

                        if nova_pos == moeda.pos:
                            capturada = True
                        if not capturada and (step % COIN_MOVE_EVERY == 0):
                            moeda.mover()
                            moeda.atualizar_cheiro()
                            if moeda.pos == nova_pos:
                                capturada = True

                        agente.pos = nova_pos

                        if capturada:
                            total_capturas += 1
                            print(f"🎯 Capturada! Total: {total_capturas}")
                            break
            except KeyboardInterrupt:
                print(f"\n\n{Fore.RED}🛑 Teste interrompido.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}📊 Total: {total_capturas}{Style.RESET_ALL}")
        else:
            raise e
    finally:
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    main()