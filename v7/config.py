# config.py - V7 (Cheiro Total + Visão em Linha Reta)

# Ambiente
MAZE_SIZE = 20
MAX_STEPS = 300

# Treinamento
EPISODES = 5000
LEARNING_RATE = 0.001
DISCOUNT = 0.95
BATCH_SIZE = 128
MEMORY_SIZE = 20000
TARGET_UPDATE = 100

# Exploração (epsilon-greedy)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Moeda
COIN_MOVE_EVERY = 1
PROB_MOEDA_PARAR = 0.15
DISTANCIA_SEGURA = 6

# Visão
VISUAL_RANGE = 8

# Cheiro – SEM LIMITE (BFS total)
# Não precisa de SMELL_RADIUS, o BFS percorre todo o labirinto

# Recompensas
CATCH_REWARD = 3.0
PROGRESS_BONUS = 0.1        # bônus por se aproximar (cheiro mais forte)
PENALIDADE_PARADO = -0.3    # penalidade por bater em parede
EXPLORATION_BONUS = 0.0     # não usado (cheiro total)

# Checkpoint
CHECKPOINT_DIR = "v7/models"
CHECKPOINT_INTERVAL = 100