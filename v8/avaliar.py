# avaliar.py - V8: mede a taxa de captura do modelo no mapa atual, sem interface
# Uso (na raiz do repo):  python v8\avaliar.py [episodios]     (padrão: 50)
# O mapa vem de $env:MAPA (e $env:SEMENTE para grande_aleatorio), como nos testes.
import os
import sys
import time
import random
import numpy as np
import torch
from config import MAX_STEPS, COIN_MOVE_EVERY, CHECKPOINT_DIR, CHECKPOINT_DIR_PADRAO, MAPA, SEMENTE
from labirinto import Labirinto, MAZE
from moeda import Moeda
from agente import Agente, encontrar_modelo


def main():
    episodios = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    random.seed(0)
    np.random.seed(0)

    mundo = Labirinto(MAZE)
    agente = Agente(mundo, torch.device("cpu"))
    caminho = encontrar_modelo(CHECKPOINT_DIR, CHECKPOINT_DIR_PADRAO)
    if caminho is None:
        sys.exit(f"Nenhum modelo encontrado em {CHECKPOINT_DIR}/ (rode na raiz do repositório)")
    ep = agente.carregar_pesos(caminho)

    nome = f"{MAPA} (semente {SEMENTE})" if MAPA == "grande_aleatorio" else MAPA
    print(f"Mapa: {nome} {mundo.tamanho}x{mundo.tamanho} | modelo: {caminho} (episódio {ep})")
    print(f"Rodando {episodios} episódios de até {MAX_STEPS} passos...", flush=True)

    capturas, passos, inicio = 0, [], time.time()
    for _ in range(episodios):
        agente.pos = mundo.posicao_vazia()
        agente.historico_posicoes.clear()
        agente.contador_loop = 0
        agente.penalidade_loop_acumulada = 0.0
        agente.prob_aleatoria = 0.0
        moeda = Moeda(mundo, agente, evitar=agente.pos)
        moeda.atualizar_cheiro()
        for step in range(MAX_STEPS):
            estado = agente.obter_estado(moeda.cheiro, moeda)
            nova_pos, _, _ = agente.tentar_mover(agente.escolher_acao(estado, 0.0))
            capturada = nova_pos == moeda.pos
            if not capturada and step % COIN_MOVE_EVERY == 0:
                moeda.mover()
                moeda.atualizar_cheiro()
                capturada = moeda.pos == nova_pos
            agente.pos = nova_pos
            if capturada:
                capturas += 1
                passos.append(step + 1)
                break

    mediana = int(np.median(passos)) if passos else "-"
    print(f"Captura: {100 * capturas / episodios:.0f}% ({capturas}/{episodios}) | "
          f"passos medianos até capturar: {mediana} | {time.time() - inicio:.0f}s")


if __name__ == "__main__":
    main()
