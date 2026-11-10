# config.py - V8 (mapas: original 20x20, grande 54x54 e variações do grande)
import os

# Mapa (variável de ambiente MAPA). Ex. (PowerShell): $env:MAPA="grande_espelho"
#   grande            54x54 da branch feat--novo-labirinto (PADRÃO)
#   grande_espelho    espelhado esquerda/direita
#   grande_vertical   de cabeça para baixo
#   grande_girado     girado 90 graus
#   grande_aleatorio  blocos abertos/fechados ao acaso; muda com $env:SEMENTE="7"
#   original          20x20 antigo
MAPA = os.environ.get("MAPA", "grande").strip().lower()
SEMENTE = int(os.environ.get("SEMENTE", "1"))
GRANDE = MAPA.startswith("grande")

# Ambiente
MAX_STEPS = 1000 if GRANDE else 300

# Treinamento
EPISODES = int(os.environ.get("EPISODIOS", "5000"))
LEARNING_RATE = 0.001
DISCOUNT = 0.95
BATCH_SIZE = 128
MEMORY_SIZE = 20000
TARGET_UPDATE = 10 if GRANDE else 100  # em episódios (no grande cada episódio tem até 1000 passos)

# Exploração
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.995
# Fine-tuning: o mapa grande começa dos pesos do modelo 20x20 (que já pega ~56% lá)
# com pouca exploração, em vez de começar do zero com epsilon 1.0.
EPSILON_FINETUNE = 0.2

# Moeda
COIN_MOVE_EVERY = 1
PROB_MOEDA_PARAR = 0.15
DISTANCIA_SEGURA = 6

# Visão (o modelo foi treinado com 8; a entrada é dist / VISUAL_RANGE)
VISUAL_RANGE = 8

# Recompensas
# Testado no fine-tuning do mapa grande (300 episódios, avaliação em 100 episódios):
#   captura 3.0 sem bônus de origem      -> 90% de captura no grande
#   captura 50.0 + bônus de origem 0.01  -> 67% (valores da branch feat--novo-labirinto)
# Para fine-tuning ficamos com os valores do treino original.
CATCH_REWARD = 3.0
PROGRESS_BONUS = 0.2
PENALIDADE_PARADO = -0.3
# Bônus por se afastar do ponto de origem (exploração). 0 = desligado.
ORIGEM_REWARD = 0.0

# Loop
LOOP_PENALTY_BASE = 0.05

# Checkpoint: todas as variações grande_* usam o mesmo modelo
CHECKPOINT_DIR = "v8/models_grande" if GRANDE else "v8/models"
CHECKPOINT_DIR_PADRAO = "v8/models"  # modelo 20x20: ponto de partida do fine-tuning e reserva dos testes
CHECKPOINT_INTERVAL = 25 if GRANDE else 100
