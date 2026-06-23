# agente.py - V7 (cheiro total + visão em linha reta)
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from rede import DQN
from config import LEARNING_RATE, BATCH_SIZE, MEMORY_SIZE, DISCOUNT, PENALIDADE_PARADO, VISUAL_RANGE
from labirinto import ACTIONS

class Agente:
    def __init__(self, labirinto, device):
        self.labirinto = labirinto
        self.device = device
        self.pos = labirinto.posicao_vazia()
        self.ultima_acao = -1
        self.passos_parados = 0

        self.policy_net = DQN(input_dim=13, output_dim=4).to(device)
        self.target_net = DQN(input_dim=13, output_dim=4).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)

    def _intensidade_cheiro(self, x, y, cheiro_matrix):
        if self.labirinto.eh_parede(x, y):
            return 0.0
        d = cheiro_matrix[x, y]
        if d == -1:
            return 0.0
        return 1.0 / (d**2 + 1)   # intensidade não-linear

    def _ver_moeda(self, moeda):
        """Retorna (dx, dy, dist) se a moeda estiver visível em linha reta sem paredes."""
        x0, y0 = self.pos
        x1, y1 = moeda.pos
        dx = x1 - x0
        dy = y1 - y0
        dist = max(abs(dx), abs(dy))
        if dist == 0 or dist > VISUAL_RANGE:
            return None
        # Verifica se está em linha reta (horizontal, vertical ou diagonal)
        if dx != 0 and dy != 0 and abs(dx) != abs(dy):
            return None
        passo_x = 1 if dx > 0 else -1 if dx < 0 else 0
        passo_y = 1 if dy > 0 else -1 if dy < 0 else 0
        x, y = x0 + passo_x, y0 + passo_y
        while (x, y) != (x1, y1):
            if self.labirinto.eh_parede(x, y):
                return None
            x += passo_x
            y += passo_y
        return (dx, dy, dist)

    def _direcao_para_entradas(self, dx, dy, dist):
        """Converte direção em 8 entradas (7 direções + distância normalizada)."""
        entradas = np.zeros(8)
        if dx == 0 and dy < 0:
            entradas[0] = dist / VISUAL_RANGE
        elif dx == 0 and dy > 0:
            entradas[1] = dist / VISUAL_RANGE
        elif dx < 0 and dy == 0:
            entradas[2] = dist / VISUAL_RANGE
        elif dx > 0 and dy == 0:
            entradas[3] = dist / VISUAL_RANGE
        elif dx < 0 and dy < 0:
            entradas[4] = dist / VISUAL_RANGE
        elif dx < 0 and dy > 0:
            entradas[5] = dist / VISUAL_RANGE
        elif dx > 0 and dy < 0:
            entradas[6] = dist / VISUAL_RANGE
        elif dx > 0 and dy > 0:
            entradas[7] = dist / VISUAL_RANGE
        return entradas

    def obter_estado(self, cheiro_matrix, moeda):
        x, y = self.pos
        # Parte cheiro (5 entradas)
        centro = self._intensidade_cheiro(x, y, cheiro_matrix)
        viz = []
        for dx, dy in ACTIONS.values():
            nx, ny = x + dx, y + dy
            viz.append(self._intensidade_cheiro(nx, ny, cheiro_matrix))
        # Parte visão (8 entradas)
        visao = self._ver_moeda(moeda)
        if visao is not None:
            dx, dy, dist = visao
            visao_entradas = self._direcao_para_entradas(dx, dy, dist)
        else:
            visao_entradas = np.zeros(8)
        estado = np.concatenate([np.array([centro] + viz), visao_entradas])
        return estado.astype(np.float32)

    def escolher_acao(self, estado, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            state_t = torch.from_numpy(estado).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def tentar_mover(self, acao):
        dx, dy = ACTIONS[acao]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if self.labirinto.eh_parede(nx, ny):
            return self.pos, True
        return [nx, ny], False

    def armazenar_transicao(self, estado, acao, recompensa, novo_estado, done):
        self.memory.append((estado, acao, recompensa, novo_estado, done))

    def aprender(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        estados = torch.from_numpy(np.array([e[0] for e in batch])).float().to(self.device)
        acoes = torch.tensor([e[1] for e in batch], dtype=torch.long).unsqueeze(1).to(self.device)
        recompensas = torch.tensor([e[2] for e in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        novos_estados = torch.from_numpy(np.array([e[3] for e in batch])).float().to(self.device)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

        q_vals = self.policy_net(estados).gather(1, acoes)
        with torch.no_grad():
            next_q = self.target_net(novos_estados).max(1, keepdim=True)[0]
            target = recompensas + DISCOUNT * next_q * (1 - dones)

        loss = F.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def atualizar_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def salvar_checkpoint(self, caminho, epsilon, episode):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': epsilon,
            'episode': episode,
            'memory': self.memory,
        }, caminho)
        print(f"Checkpoint salvo em {caminho}")

    def carregar_checkpoint(self, caminho):
        checkpoint = torch.load(caminho, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
        episode = checkpoint['episode']
        self.memory = checkpoint.get('memory', deque(maxlen=MEMORY_SIZE))
        return epsilon, episode