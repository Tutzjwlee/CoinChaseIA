# moeda.py - V5 (cheiro em todo o labirinto)
import numpy as np
import random
from collections import deque
from labirinto import ACTIONS
from config import PROB_MOEDA_PARAR, DISTANCIA_SEGURA

class Moeda:
    def __init__(self, labirinto, cacador, evitar=None):
        self.labirinto = labirinto
        self.cacador = cacador
        self.pos = labirinto.posicao_vazia()
        while evitar is not None and self.pos == evitar:
            self.pos = labirinto.posicao_vazia()
        self.direcao = labirinto.direcao_aleatoria(self.pos)
        self.cheiro = None
        self.passos_parados = 0

    def _bfs_cheiro(self):
        """
        Calcula a distância de cada célula até a moeda SEM LIMITE DE ALCANCE.
        O cheiro percorre TODO o labirinto.
        """
        dist = np.full((self.labirinto.tamanho, self.labirinto.tamanho), -1, dtype=int)
        q = deque()
        x, y = self.pos
        dist[x, y] = 0
        q.append((x, y))
        while q:
            cx, cy = q.popleft()
            for dx, dy in ACTIONS.values():
                nx, ny = cx + dx, cy + dy
                if not self.labirinto.eh_parede(nx, ny) and dist[nx, ny] == -1:
                    dist[nx, ny] = dist[cx, cy] + 1
                    q.append((nx, ny))
        return dist

    def atualizar_cheiro(self):
        """Atualiza o cheiro com alcance TOTAL (labirinto inteiro)"""
        self.cheiro = self._bfs_cheiro()  # sem limite de alcance!

    def mover(self):
        """Moeda foge do caçador, pode parar e é mais lenta"""
        if random.random() < PROB_MOEDA_PARAR:
            self.passos_parados += 1
            return

        cx, cy = self.cacador.pos
        dist_atual = abs(self.pos[0] - cx) + abs(self.pos[1] - cy)

        if dist_atual < DISTANCIA_SEGURA:
            nova_direcao = self._escolher_direcao_fuga()
            if nova_direcao != -1:
                self.direcao = nova_direcao
        else:
            if random.random() < 0.3:
                self.passos_parados += 1
                return
            opcoes = self.labirinto.direcoes_validas(self.pos)
            if opcoes:
                self.direcao = random.choice(opcoes)
            else:
                return

        dx, dy = ACTIONS[self.direcao]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if self.labirinto.eh_parede(nx, ny):
            if dist_atual < DISTANCIA_SEGURA:
                nova_direcao = self._escolher_direcao_fuga()
            else:
                opcoes = self.labirinto.direcoes_validas(self.pos)
                nova_direcao = random.choice(opcoes) if opcoes else -1
            if nova_direcao != -1:
                self.direcao = nova_direcao
                dx, dy = ACTIONS[self.direcao]
                nx, ny = self.pos[0] + dx, self.pos[1] + dy
                if self.labirinto.eh_parede(nx, ny):
                    return
            else:
                return

        self.pos = [nx, ny]
        self.passos_parados = 0

    def _escolher_direcao_fuga(self):
        """Escolhe direção que maximiza distância do caçador"""
        opcoes = self.labirinto.direcoes_validas(self.pos)
        if not opcoes:
            return -1
        cx, cy = self.cacador.pos
        melhor_dist = -1
        melhores = []
        for acao in opcoes:
            dx, dy = ACTIONS[acao]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            dist = abs(nx - cx) + abs(ny - cy)
            if dist > melhor_dist:
                melhor_dist = dist
                melhores = [acao]
            elif dist == melhor_dist:
                melhores.append(acao)
        return random.choice(melhores) if melhores else -1