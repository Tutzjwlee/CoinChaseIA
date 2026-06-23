# main.py - V5 (corrigido)
import os
import time
import numpy as np
import torch
from IPython.display import clear_output

# CORRIGIDO: import sem prefixo v5
from config import (
    EPISODES, MAX_STEPS, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    COIN_MOVE_EVERY, CATCH_REWARD, PROGRESS_BONUS, PENALIDADE_PARADO,
    CHECKPOINT_DIR, CHECKPOINT_INTERVAL, TARGET_UPDATE
)
from labirinto import Labirinto, MAZE
from moeda import Moeda
from agente import Agente
from utils import render, plotar_taxa_captura

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    print(f"Versão: V5 - Visão em Linha Reta")
    print(f"Checkpoints salvos em: {CHECKPOINT_DIR}/")

    mundo = Labirinto(MAZE)
    agente = Agente(mundo, device)

    # Cria a pasta v5/models se não existir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
    epsilon = EPSILON_START
    start_episode = 0
    
    if os.path.exists(latest_checkpoint):
        print("Carregando checkpoint existente...")
        epsilon, start_episode = agente.carregar_checkpoint(latest_checkpoint)
        print(f"Retomando do episódio {start_episode}, epsilon={epsilon:.3f}")

    catches_history = []
    rewards_history = []

    for episode in range(start_episode, EPISODES):
        agente.pos = mundo.posicao_vazia()
        agente.passos_parados = 0
        agente.repeticoes = 0
        moeda = Moeda(mundo, agente, evitar=agente.pos)
        moeda.atualizar_cheiro()

        total_reward = 0.0
        caught = False

        for step in range(MAX_STEPS):
            estado = agente.obter_estado(moeda.cheiro, moeda)
            acao = agente.escolher_acao(estado, epsilon)
            nova_pos, bateu = agente.tentar_mover(acao)

            reward = 0.0

            # Penalidade por bater na parede
            if bateu:
                reward += PENALIDADE_PARADO
                agente.passos_parados += 1
                if agente.passos_parados > 3:
                    reward += PENALIDADE_PARADO * 0.5
            else:
                agente.passos_parados = 0

            # Penalidade por repetir ação (loop)
            if agente.repeticoes > 3:
                reward += PENALIDADE_PARADO * 0.5

            # Captura pelo movimento do agente
            if nova_pos == moeda.pos:
                reward += CATCH_REWARD
                caught = True

            # Movimento da moeda
            if not caught and (step % COIN_MOVE_EVERY == 0):
                moeda.mover()
                moeda.atualizar_cheiro()
                if moeda.pos == nova_pos:
                    reward += CATCH_REWARD
                    caught = True

            # Bônus de progresso (cheiro mais forte)
            cheiro_anterior = agente._intensidade_cheiro(agente.pos[0], agente.pos[1], moeda.cheiro)
            agente.pos = nova_pos
            novo_estado = agente.obter_estado(moeda.cheiro, moeda)
            cheiro_novo = agente._intensidade_cheiro(nova_pos[0], nova_pos[1], moeda.cheiro)
            if cheiro_novo > cheiro_anterior:
                reward += PROGRESS_BONUS

            done = caught or (step == MAX_STEPS - 1)
            agente.armazenar_transicao(estado, acao, reward, novo_estado, done)
            agente.aprender()

            total_reward += reward
            if caught:
                break

        if episode % TARGET_UPDATE == 0:
            agente.atualizar_target()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        catches_history.append(1 if caught else 0)
        rewards_history.append(total_reward)

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{episode+1}.pth")
            agente.salvar_checkpoint(checkpoint_path, epsilon, episode+1)
            agente.salvar_checkpoint(latest_checkpoint, epsilon, episode+1)

        if (episode + 1) % 100 == 0:
            taxa = 100 * np.mean(catches_history[-100:])
            print(f"Episódio {episode+1} | Captura (últ.100): {taxa:.1f}% | Epsilon: {epsilon:.3f}")

    print("Treinamento concluído!")
    agente.salvar_checkpoint(os.path.join(CHECKPOINT_DIR, "checkpoint_final.pth"), epsilon, EPISODES)

    # Teste visual
    print("\n--- Teste visual (política greedy) ---")
    agente.pos = mundo.posicao_vazia()
    moeda = Moeda(mundo, agente, evitar=agente.pos)
    moeda.atualizar_cheiro()

    for step in range(MAX_STEPS):
        clear_output(wait=True)
        estado = agente.obter_estado(moeda.cheiro, moeda)
        acao = agente.escolher_acao(estado, 0.0)
        nova_pos, _ = agente.tentar_mover(acao)

        caught = (nova_pos == moeda.pos)
        if not caught and (step % COIN_MOVE_EVERY == 0):
            moeda.mover()
            moeda.atualizar_cheiro()
            if moeda.pos == nova_pos:
                caught = True

        agente.pos = nova_pos
        render(mundo, agente.pos, moeda.pos, moeda.cheiro, "TESTE", step)
        if caught:
            print("Moeda capturada!")
            break
        time.sleep(0.2)

    plotar_taxa_captura(catches_history)

if __name__ == "__main__":
    main()