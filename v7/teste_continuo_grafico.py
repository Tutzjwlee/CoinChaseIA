# teste_continuo_grafico.py - V7 (Otimizado)
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg' se tiver PyQt5
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import sys
from config import MAX_STEPS, COIN_MOVE_EVERY, CHECKPOINT_DIR
from labirinto import Labirinto, MAZE
from moeda import Moeda
from agente import Agente
from colorama import Fore, Style

# Configurações de plot
plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))
fig.canvas.manager.set_window_title('🐺 Hunter vs Coin 🪙')
plt.tight_layout()

# Variável para controlar se a janela está aberta
janela_aberta = True

def on_close(event):
    global janela_aberta
    janela_aberta = False
    print("\n🔴 Janela fechada. O teste continua no terminal...")

fig.canvas.mpl_connect('close_event', on_close)

def render_plot(labirinto, pos_hunter, pos_coin, cheiro, total_capturas, step, visao_ativa, distancia_visao):
    """Renderiza o labirinto com mapa de calor do cheiro (atualização a cada 2 passos)."""
    if not janela_aberta:
        return

    ax.clear()
    height, width = labirinto.tamanho, labirinto.tamanho
    heatmap = np.zeros((height, width))

    if cheiro is not None:
        for i in range(height):
            for j in range(width):
                if cheiro[i, j] != -1:
                    heatmap[i, j] = 1.0 / (cheiro[i, j]**2 + 1)

    # Mapa de calor (cores mais vivas)
    im = ax.imshow(heatmap, cmap='viridis', vmin=0, vmax=1, origin='upper')

    # Paredes
    for i in range(height):
        for j in range(width):
            if labirinto.eh_parede(i, j):
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color='#2c3e50', lw=0))

    # Hunter (quadrado vermelho)
    hx, hy = pos_hunter
    ax.add_patch(plt.Rectangle((hy-0.4, hx-0.4), 0.8, 0.8, color='red', ec='darkred', lw=2))

    # Coin (círculo amarelo)
    cx, cy = pos_coin
    ax.add_patch(plt.Circle((cy, cx), 0.4, color='gold', ec='orange', lw=2))

    # Título com informações
    titulo = f"Moedas: {total_capturas} | Passo: {step}"
    if visao_ativa:
        titulo += f" | 👁️ Visão ativa! Dist: {distancia_visao}"
    else:
        titulo += " | 👃 Olfato"
    ax.set_title(titulo, fontsize=11, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(height-0.5, -0.5)

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)  # pausa muito curta para manter responsivo

def main():
    global janela_aberta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    print(f"Versão: V7 - Teste Gráfico (Otimizado)")

    mundo = Labirinto(MAZE)
    agente = Agente(mundo, device)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
    if os.path.exists(checkpoint_path):
        try:
            agente.carregar_checkpoint(checkpoint_path)
            print(f"Modelo carregado: {checkpoint_path}")
        except Exception as e:
            print(f"Erro: {e}. Tentando checkpoint_final...")
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

    print("\n" + "="*50)
    print(f"{Style.BRIGHT}🐺 HUNTER vs COIN 🪙 (Modo Gráfico Otimizado)")
    print("="*50)
    print("A janela gráfica será aberta. Você pode fechá-la a qualquer momento.")
    print("O teste continuará no terminal até você pressionar Ctrl+C.\n")

    total_capturas = 0
    try:
        while True:
            agente.pos = mundo.posicao_vazia()
            moeda = Moeda(mundo, agente, evitar=agente.pos)
            moeda.atualizar_cheiro()

            capturada = False
            for step in range(MAX_STEPS):
                # Atualiza o gráfico a cada 2 passos para ganhar velocidade
                if step % 2 == 0 and janela_aberta:
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

                    visao = agente._ver_moeda(moeda)
                    visao_ativa = visao is not None
                    distancia_visao = visao[2] if visao_ativa else None

                    render_plot(
                        mundo,
                        agente.pos,
                        moeda.pos,
                        moeda.cheiro,
                        total_capturas,
                        step,
                        visao_ativa,
                        distancia_visao
                    )

                    if capturada:
                        total_capturas += 1
                        print(f"🎯 Capturada! Total: {total_capturas}")
                        time.sleep(0.2)
                        break
                else:
                    # Fora da atualização gráfica, apenas processa o movimento
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
                    if capturada:
                        total_capturas += 1
                        print(f"🎯 Capturada! Total: {total_capturas}")
                        time.sleep(0.2)
                        break
            # Se a janela foi fechada, continuamos rodando sem gráfico
            if not janela_aberta:
                # Modo terminal apenas (sem gráfico)
                # (Já estamos fazendo isso, só continua)
                pass

    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}🛑 Teste interrompido pelo usuário.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 Total de moedas capturadas: {total_capturas}{Style.RESET_ALL}")
    except Exception as e:
        # Captura qualquer erro (ex: fechamento da janela)
        if "closed" in str(e).lower() or "disconnected" in str(e).lower():
            print("\n🔴 Janela fechada. Teste continua no terminal...")
            # Continua o loop sem gráfico
            # (Já estamos processando, mas precisamos continuar sem render)
            # Vamos refazer o loop principal sem renderização
            try:
                while True:
                    agente.pos = mundo.posicao_vazia()
                    moeda = Moeda(mundo, agente, evitar=agente.pos)
                    moeda.atualizar_cheiro()
                    capturada = False
                    for step in range(MAX_STEPS):
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