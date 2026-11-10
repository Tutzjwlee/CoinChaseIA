# visualizar.py - V8: visualização da demo (matplotlib), com troca de mapa ao vivo
#
# Uso (na raiz do repo):
#   python v8\visualizar.py                      # mapa de $env:MAPA (padrão: grande)
#   python v8\visualizar.py --mapa grande_girado
#   python v8\visualizar.py --modelo v8\models\checkpoint_latest.pth --visao 16 --passos 300 --titulo "Branch feat--novo-labirinto"
#
# Teclas: espaço = pausar · + / - = velocidade · m = próximo mapa · r = novo mapa aleatório
#         c = mostrar/esconder cheiro · t = rastro · q = sair
import argparse
import os
import random
import sys
import time
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch

import config
import labirinto as lab
import agente as agente_mod
from labirinto import Labirinto
from moeda import Moeda
from agente import Agente

# ---------------------------------------------------------------- aparência
COR_FUNDO = "#13201D"
COR_PAREDE = np.array([0x0B, 0x13, 0x11]) / 255
COR_LONGE = np.array([0x22, 0x3A, 0x35]) / 255    # chão longe da moeda
COR_PERTO = np.array([0xE0, 0xA8, 0x2E]) / 255    # chão perto da moeda
COR_CHAO_SEM_CHEIRO = np.array([0x2A, 0x3D, 0x39]) / 255
COR_CACADOR = "#F06A4F"
COR_MOEDA = "#FFD34D"
COR_VISAO = "#7FD1FF"
COR_TEXTO = "#F3EFE4"
COR_TEXTO_2 = "#9FB5AF"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

MAPAS_CICLO = ["grande", "grande_espelho", "grande_vertical", "grande_girado", "grande_aleatorio"]
NOMES = {
    "original": "Original 20x20",
    "grande": "Grande 54x54",
    "grande_espelho": "Grande · espelhado",
    "grande_vertical": "Grande · invertido",
    "grande_girado": "Grande · girado 90°",
    "grande_aleatorio": "Grande · aleatório",
}
VELOCIDADES = [1, 2, 4, 8, 16]   # passos de simulação por quadro
ESCALA = 1.0                      # tamanho da janela (1.0 = 14 polegadas de largura)


def F(tamanho):
    return tamanho * ESCALA


def caminho_modelo(nome_mapa, forcado=None):
    if forcado:
        return forcado
    pasta = "v8/models" if nome_mapa == "original" else "v8/models_grande"
    caminho = os.path.join(pasta, "checkpoint_latest.pth")
    if not os.path.exists(caminho):
        caminho = os.path.join(config.CHECKPOINT_DIR_PADRAO, "checkpoint_latest.pth")
    return caminho


def grade_do_mapa(nome, semente):
    if nome == "grande_aleatorio":
        return lab.variacao_aleatoria(lab.MAZE_GRANDE, semente)
    return lab.MAPAS[nome]


class Demo:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        agente_mod.VISUAL_RANGE = args.visao          # usado em obter_estado / _ver_moeda
        self.max_passos = args.passos
        self.nome_mapa = args.mapa
        self.semente = args.semente
        self.pausado = False
        self.vel_idx = 1
        self.mostrar_cheiro = True
        self.mostrar_rastro = True
        self.modelo_carregado = None

        self.capturas = 0
        self.episodios = 0
        self.passos_capturas = deque(maxlen=50)
        self.mensagem = ""
        self.mensagem_ate = 0.0

        self._montar_figura()
        self._carregar_mapa(self.nome_mapa)

    # ------------------------------------------------------------ figura
    def _montar_figura(self):
        self.fig = plt.figure(figsize=(14 * ESCALA, 8.4 * ESCALA), facecolor=COR_FUNDO)
        titulo = self.args.titulo or "CoinChaseIA"
        self.fig.canvas.manager.set_window_title(titulo)
        if self.args.posicao:
            try:
                self.fig.canvas.manager.window.wm_geometry(self.args.posicao)
            except Exception:
                pass
        self.ax = self.fig.add_axes([0.02, 0.03, 0.60, 0.94])
        self.ax.set_facecolor(COR_FUNDO)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for s in self.ax.spines.values():
            s.set_visible(False)
        self.painel = self.fig.add_axes([0.65, 0.03, 0.33, 0.94])
        self.painel.axis("off")

        self.img = self.ax.imshow(np.zeros((2, 2, 3)), interpolation="nearest", origin="upper")
        self.rastro = LineCollection([], linewidths=3, capstyle="round")
        self.ax.add_collection(self.rastro)
        (self.linha_visao,) = self.ax.plot([], [], color=COR_VISAO, lw=2.5, ls=(0, (4, 3)), alpha=0.95)
        self.moeda_pt = self.ax.scatter([], [], s=1, c=COR_MOEDA, edgecolors="#6B4600", linewidths=1.5, zorder=5)
        self.cacador_pt = self.ax.scatter([], [], s=1, c=COR_CACADOR, edgecolors="white", linewidths=1.8, zorder=6)
        self.aviso = self.ax.text(0.5, 0.5, "", transform=self.ax.transAxes, ha="center", va="center",
                                  fontsize=F(34), fontweight="bold", color=COR_TEXTO, zorder=10,
                                  bbox=dict(boxstyle="round,pad=0.5", fc="#13201DDD", ec="none"))
        self.aviso.set_visible(False)

        p = self.painel
        self.t_titulo = p.text(0, 0.97, titulo, fontsize=F(24), fontweight="bold", color=COR_TEXTO, va="top")
        self.t_mapa = p.text(0, 0.905, "", fontsize=F(15), color=COR_TEXTO_2, va="top")
        self.t_modelo = p.text(0, 0.865, "", fontsize=F(12), color=COR_TEXTO_2, va="top")
        p.text(0, 0.78, "MOEDAS", fontsize=F(12), color=COR_TEXTO_2, va="top", fontweight="bold")
        self.t_capturas = p.text(0, 0.755, "0", fontsize=F(54), fontweight="bold", color=COR_MOEDA, va="top")
        p.text(0.5, 0.78, "TAXA DE CAPTURA", fontsize=F(12), color=COR_TEXTO_2, va="top", fontweight="bold")
        self.t_taxa = p.text(0.5, 0.755, "–", fontsize=F(54), fontweight="bold", color=COR_TEXTO, va="top")
        self.t_info = p.text(0, 0.58, "", fontsize=F(15), color=COR_TEXTO, va="top", linespacing=1.7)
        self.t_modo = p.text(0, 0.40, "", fontsize=F(17), fontweight="bold", va="top",
                             bbox=dict(boxstyle="round,pad=0.45", fc="#223A35", ec="none"))
        p.text(0, 0.30, "LEGENDA", fontsize=F(12), color=COR_TEXTO_2, va="top", fontweight="bold")
        p.scatter([0.02], [0.245], s=F(160) * ESCALA, c=COR_CACADOR, edgecolors="white", linewidths=1.5, transform=p.transAxes)
        p.text(0.06, 0.245, "caçador (agente)", fontsize=F(13), color=COR_TEXTO, va="center")
        p.scatter([0.52], [0.245], s=F(160) * ESCALA, c=COR_MOEDA, edgecolors="#6B4600", linewidths=1.5, transform=p.transAxes)
        p.text(0.56, 0.245, "moeda", fontsize=F(13), color=COR_TEXTO, va="center")
        grad = np.linspace(0, 1, 100)[None, :, None]
        barra = COR_LONGE + (COR_PERTO - COR_LONGE) * grad ** 2.0
        p.imshow(barra, extent=(0.0, 0.6, 0.175, 0.205), aspect="auto", transform=p.transAxes)
        p.text(0.0, 0.165, "cheiro: longe", fontsize=F(11), color=COR_TEXTO_2, va="top")
        p.text(0.6, 0.165, "perto", fontsize=F(11), color=COR_TEXTO_2, va="top", ha="right")
        p.text(0, 0.10, "espaço pausar · +/- velocidade · m próximo mapa\nr mapa aleatório novo · c cheiro · t rastro · q sair",
               fontsize=F(11), color=COR_TEXTO_2, va="top", linespacing=1.6)
        p.set_xlim(0, 1)
        p.set_ylim(0, 1)

        self.fig.canvas.mpl_connect("key_press_event", self._tecla)

    # ------------------------------------------------------------ mapa / modelo
    def _carregar_mapa(self, nome):
        self.nome_mapa = nome
        self.mundo = Labirinto(grade_do_mapa(nome, self.semente))
        n = self.mundo.tamanho
        self.paredes = self.mundo.grade == 1
        self.ax.set_xlim(-0.5, n - 0.5)
        self.ax.set_ylim(n - 0.5, -0.5)
        self.img.set_extent((-0.5, n - 0.5, n - 0.5, -0.5))
        casa_pt = 0.94 * 8.4 * ESCALA * 72 / n          # lado de uma casa em pontos
        self.cacador_pt.set_sizes([max(casa_pt * 1.15, F(15)) ** 2])
        self.moeda_pt.set_sizes([max(casa_pt * 1.0, F(13)) ** 2])

        caminho = caminho_modelo(nome, self.args.modelo)
        if self.modelo_carregado != caminho or not hasattr(self, "agente"):
            self.agente = Agente(self.mundo, self.device)
            _, self.ep_modelo = self.agente.carregar_checkpoint(caminho)
            self.agente.memory.clear()
            self.modelo_carregado = caminho
        self.agente.labirinto = self.mundo
        rotulo = NOMES.get(nome, nome) + (f" (semente {self.semente})" if nome == "grande_aleatorio" else "")
        self.t_mapa.set_text(f"{rotulo} · até {self.max_passos} passos")
        pasta = os.path.basename(os.path.dirname(caminho))
        self.t_modelo.set_text(f"modelo: {pasta} · episódio {self.ep_modelo} · visão {self.args.visao}")
        self.capturas = self.episodios = 0
        self.passos_capturas.clear()
        self._novo_episodio()

    def _novo_episodio(self):
        a = self.agente
        a.pos = self.mundo.posicao_vazia()
        a.historico_posicoes.clear()
        a.contador_loop = 0
        a.penalidade_loop_acumulada = 0.0
        a.prob_aleatoria = 0.0
        self.moeda = Moeda(self.mundo, a, evitar=a.pos)
        self.moeda.atualizar_cheiro()
        self.passo = 0
        self.trilha = deque([tuple(a.pos)], maxlen=40)

    # ------------------------------------------------------------ simulação
    def _passo(self):
        a, m = self.agente, self.moeda
        estado = a.obter_estado(m.cheiro, m)
        nova_pos, _, _ = a.tentar_mover(a.escolher_acao(estado, 0.0))
        capturada = nova_pos == m.pos
        if not capturada and self.passo % config.COIN_MOVE_EVERY == 0:
            m.mover()
            m.atualizar_cheiro()
            capturada = m.pos == nova_pos
        a.pos = nova_pos
        self.trilha.append(tuple(nova_pos))
        self.passo += 1
        if capturada:
            self.capturas += 1
            self.episodios += 1
            self.passos_capturas.append(self.passo)
            self._avisar(f"Capturada em {self.passo} passos!", 0.9)
            return True
        if self.passo >= self.max_passos:
            self.episodios += 1
            self._avisar("Tempo esgotado", 0.9)
            return True
        return False

    def _avisar(self, texto, segundos):
        self.mensagem = texto
        self.mensagem_ate = time.time() + segundos

    # ------------------------------------------------------------ desenho
    def _desenhar(self):
        n = self.mundo.tamanho
        rgb = np.empty((n, n, 3))
        if self.mostrar_cheiro:
            d = self.moeda.cheiro.astype(float)
            dmax = d.max() if d.max() > 0 else 1.0
            prox = np.clip(1.0 - d / dmax, 0, 1) ** 2.0     # 1 = em cima da moeda, 0 = mais longe
            rgb[:] = COR_LONGE + (COR_PERTO - COR_LONGE) * prox[..., None]
        else:
            rgb[:] = COR_CHAO_SEM_CHEIRO
        rgb[self.paredes] = COR_PAREDE
        self.img.set_data(rgb)

        hx, hy = self.agente.pos
        cx, cy = self.moeda.pos
        self.cacador_pt.set_offsets([[hy, hx]])
        self.moeda_pt.set_offsets([[cy, cx]])

        pts = [(y, x) for x, y in self.trilha]
        if self.mostrar_rastro and len(pts) > 1:
            segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
            alfas = np.linspace(0.05, 0.85, len(segs))
            cores = np.zeros((len(segs), 4))
            cores[:, :3] = matplotlib.colors.to_rgb(COR_CACADOR)
            cores[:, 3] = alfas
            self.rastro.set_segments(segs)
            self.rastro.set_color(cores)
        else:
            self.rastro.set_segments([])

        visao = self.agente._ver_moeda(self.moeda)
        if visao:
            self.linha_visao.set_data([hy, cy], [hx, cx])
            self.t_modo.set_text(f"  VISÃO · moeda a {visao[2]} casas  ")
            self.t_modo.set_color(COR_VISAO)
        else:
            self.linha_visao.set_data([], [])
            dist = int(self.moeda.cheiro[hx, hy])
            self.t_modo.set_text(f"  OLFATO · {dist} passos até a moeda  ")
            self.t_modo.set_color(COR_MOEDA)

        self.t_capturas.set_text(str(self.capturas))
        self.t_taxa.set_text(f"{100 * self.capturas / self.episodios:.0f}%" if self.episodios else "–")
        mediana = int(np.median(self.passos_capturas)) if self.passos_capturas else "–"
        vel = VELOCIDADES[self.vel_idx]
        estado = "PAUSADO" if self.pausado else f"velocidade {vel}x"
        self.t_info.set_text(
            f"Episódio {self.episodios + 1}   ·   passo {self.passo}/{self.max_passos}\n"
            f"Passos medianos até capturar: {mediana}\n"
            f"{estado}"
        )

        if time.time() < self.mensagem_ate:
            self.aviso.set_text(self.mensagem)
            self.aviso.set_visible(True)
        else:
            self.aviso.set_visible(False)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ teclado
    def _tecla(self, ev):
        k = ev.key
        if k == " ":
            self.pausado = not self.pausado
        elif k in ("+", "="):
            self.vel_idx = min(len(VELOCIDADES) - 1, self.vel_idx + 1)
        elif k == "-":
            self.vel_idx = max(0, self.vel_idx - 1)
        elif k == "m":
            atual = self.nome_mapa if self.nome_mapa in MAPAS_CICLO else MAPAS_CICLO[-1]
            self._carregar_mapa(MAPAS_CICLO[(MAPAS_CICLO.index(atual) + 1) % len(MAPAS_CICLO)])
            self._avisar(NOMES[self.nome_mapa], 1.2)
        elif k == "r":
            self.semente += 1
            self._carregar_mapa("grande_aleatorio")
            self._avisar(f"Mapa aleatório · semente {self.semente}", 1.2)
        elif k == "c":
            self.mostrar_cheiro = not self.mostrar_cheiro
        elif k == "t":
            self.mostrar_rastro = not self.mostrar_rastro
        elif k == "q":
            plt.close(self.fig)

    # ------------------------------------------------------------ laço
    def rodar(self):
        plt.show(block=False)
        pausa_fim = 0.0
        while plt.fignum_exists(self.fig.number):
            agora = time.time()
            if not self.pausado and agora >= pausa_fim:
                for _ in range(VELOCIDADES[self.vel_idx]):
                    if self._passo():
                        self._desenhar()
                        plt.pause(0.7)          # mostra a captura antes do próximo episódio
                        self._novo_episodio()
                        break
            self._desenhar()
            plt.pause(0.03)
        print(f"Encerrado. Moedas: {self.capturas} em {self.episodios} episódios.")


def main():
    p = argparse.ArgumentParser(description="Visualização da demo do CoinChaseIA")
    p.add_argument("--mapa", default=config.MAPA, choices=list(lab.MAPAS))
    p.add_argument("--semente", type=int, default=config.SEMENTE)
    p.add_argument("--modelo", default=None, help="checkpoint .pth (padrão: o do mapa)")
    p.add_argument("--visao", type=int, default=config.VISUAL_RANGE)
    p.add_argument("--passos", type=int, default=config.MAX_STEPS)
    p.add_argument("--titulo", default=None)
    p.add_argument("--tamanho", type=float, default=1.0, help="escala da janela (ex.: 0.65 para duas lado a lado)")
    p.add_argument("--posicao", default=None, help="posição da janela na tela, ex.: +0+0")
    args = p.parse_args()
    global ESCALA
    ESCALA = args.tamanho
    random.seed()
    Demo(args).rodar()


if __name__ == "__main__":
    main()
