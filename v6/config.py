# config.py - V6.6 (Exploração Aumentada)

# Ambiente
MAZE_SIZE = 20
MAX_STEPS = 300

# Treinamento - 20.000 EPISÓDIOS
EPISODES = 20000
LEARNING_RATE = 0.001
DISCOUNT = 0.95
BATCH_SIZE = 128
MEMORY_SIZE = 30000
TARGET_UPDATE = 100

# Exploração (MAIS ALEATÓRIA)
EPSILON_START = 1.0
EPSILON_END = 0.1              # ↑ 10% de exploração constante
EPSILON_DECAY = 0.997          # ↓ decaimento mais lento (explora por mais tempo)

# Moeda
COIN_MOVE_EVERY = 2
PROB_MOEDA_PARAR = 0.15
DISTANCIA_SEGURA = 6

# Visão
VISUAL_RANGE = 8

# Cheiro
SMELL_RADIUS = 15              # ↑ raio maior para dar mais pistas

# Recompensas
CATCH_REWARD = 3.0
PROGRESS_BONUS = 0.15          # ↑ incentiva mais a aproximação
EXPLORATION_BONUS = 0.8        # ↑ incentiva mais a exploração de novas células
STEP_REWARD = 0.005
PENALIDADE_PARADO = -0.3

# Anti-loop (leve)
REPETITION_PENALTY = -0.09     # ↓ penalidade mais suave

# Checkpoint
CHECKPOINT_DIR = "v6/models"
CHECKPOINT_INTERVAL = 200

# Ciclos de exploração (resetar epsilon a cada N episódios)
EXPLORATION_CYCLE = 1000
EXPLORATION_RESET = 0.5        # ↑ 50% de exploração no reset