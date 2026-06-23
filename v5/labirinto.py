# labirinto.py - V5
import numpy as np
import random

MAZE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,0,1],
    [1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1],
    [1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1],
    [1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1],
    [1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])

ACTIONS = {
    0: (-1, 0),  # cima
    1: (1, 0),   # baixo
    2: (0, -1),  # esquerda
    3: (0, 1)    # direita
}

class Labirinto:
    def __init__(self, grade=MAZE):
        self.grade = grade
        self.tamanho = len(grade)
        self.livres = [(i, j) for i in range(self.tamanho) for j in range(self.tamanho) if grade[i][j] == 0]

    def eh_parede(self, x, y):
        if x < 0 or y < 0 or x >= self.tamanho or y >= self.tamanho:
            return True
        return self.grade[x][y] == 1

    def posicao_vazia(self):
        return list(random.choice(self.livres))

    def direcoes_validas(self, pos):
        x, y = pos
        return [a for a in ACTIONS if not self.eh_parede(x + ACTIONS[a][0], y + ACTIONS[a][1])]

    def direcao_aleatoria(self, pos):
        opcoes = self.direcoes_validas(pos)
        return random.choice(opcoes) if opcoes else -1