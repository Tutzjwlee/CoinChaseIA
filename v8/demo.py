# demo.py - V8: abre a demo da aula com duas janelas lado a lado
#   esquerda: labirinto 20x20 (modelo base) · direita: labirinto 54x54 (modelo ajustado)
#
# Uso (na raiz do repo):  python v8\demo.py
#   --direita grande_aleatorio   muda o mapa da janela da direita
#   --tamanho 0.6                força o tamanho das janelas (padrão: calculado pela tela)
#   --so-mostrar                 só mostra o tamanho calculado, sem abrir nada
# Ctrl+C neste terminal fecha as duas janelas.
import argparse
import ctypes
import os
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LARGURA_BASE, ALTURA_BASE = 1400, 840   # tamanho da janela do visualizar.py com --tamanho 1.0 (a 96 dpi)


def medir_tela():
    """Largura/altura da tela em pixels reais e o fator de escala do Windows (1.0 = 100%)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # mesmas unidades que o matplotlib usa
    except Exception:
        pass
    import tkinter as tk
    raiz = tk.Tk()
    raiz.withdraw()
    largura, altura = raiz.winfo_screenwidth(), raiz.winfo_screenheight()
    fator = round(raiz.winfo_fpixels("1i") / 96, 2)
    raiz.destroy()
    return largura, altura, fator


def main():
    p = argparse.ArgumentParser(description="Demo lado a lado: 20x20 e 54x54")
    p.add_argument("--direita", default="grande", help="mapa da janela da direita (padrão: grande)")
    p.add_argument("--tamanho", type=float, default=None)
    p.add_argument("--so-mostrar", action="store_true")
    args = p.parse_args()

    largura, altura, fator = medir_tela()
    if args.tamanho:
        tamanho = args.tamanho
    else:
        cabe_largura = (largura / 2 - 12) / (LARGURA_BASE * fator)
        cabe_altura = (altura - 100 * fator) / (ALTURA_BASE * fator)   # barra de título + barra de tarefas
        tamanho = max(0.4, min(1.0, cabe_largura, cabe_altura))
    print(f"Tela {largura}x{altura} (escala {fator:.0%}) -> janelas com --tamanho {tamanho:.2f}")
    if args.so_mostrar:
        return

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    script = os.path.join("v8", "visualizar.py")
    janelas = [
        ["--mapa", "original", "--posicao", "+0+0", "--titulo", "CoinChaseIA · 20x20"],
        ["--mapa", args.direita, "--posicao", f"+{largura // 2}+0", "--titulo", "CoinChaseIA · 54x54"],
    ]
    processos = [
        subprocess.Popen([sys.executable, script, "--tamanho", f"{tamanho:.2f}"] + extra, cwd=RAIZ, env=env)
        for extra in janelas
    ]
    print("Duas janelas abertas. Teclas: m troca o mapa · r mapa aleatório · c cheiro · espaço pausa · +/- velocidade · q fecha")
    try:
        for proc in processos:
            proc.wait()
    except KeyboardInterrupt:
        for proc in processos:
            proc.terminate()
        print("Demo encerrada.")


if __name__ == "__main__":
    main()
