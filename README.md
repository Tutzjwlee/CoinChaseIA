# CoinChaseIA

Um caçador aprende sozinho, com aprendizado por reforço (DQN), a capturar uma moeda que foge dentro de um labirinto. A versão atual é a **v8**, que roda em dois mapas: o **20x20** original e um **54x54** com variações.

| Mapa | Modelo | Captura* |
|---|---|---|
| 20x20 | `v8/models/` (1200 episódios de treino) | 100% |
| 54x54 | `v8/models_grande/` (modelo do 20x20 + 300 episódios de ajuste no 54x54) | 90% |
| Variações do 54x54 (nunca vistas no treino) | o mesmo de `v8/models_grande/` | 77–93% |

\* 100 episódios por mapa, sem exploração aleatória.

---

## 1. Instalar

Precisa de **Python 3.10 ou mais novo**. Na pasta do repositório (`CoinChaseIA`), no PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Se o PowerShell bloquear o `Activate.ps1`, rode antes: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
>
> No Linux/macOS: `source .venv/bin/activate`, troque `\` por `/` nos caminhos e instale o Tkinter se faltar (`sudo apt install python3-tk`).

## 2. Rodar a demo (20x20 e 54x54 lado a lado)

```powershell
python v8\demo.py
```

Ou dê **dois cliques em `demo_aula.bat`**. As duas janelas se ajustam ao tamanho da tela. **Ctrl+C** no terminal fecha as duas.

**Teclas** (com a janela selecionada):

| Tecla | O que faz |
|---|---|
| `m` | troca o mapa: grande → espelhado → invertido → girado → aleatório |
| `r` | gera um mapa aleatório novo |
| `c` | liga/desliga o cheiro |
| `t` | liga/desliga o rastro do caçador |
| `espaço` | pausa |
| `+` / `-` | velocidade |
| `q` | fecha a janela |

**Na tela:** o degradê dourado é o cheiro da moeda (mais claro = mais perto; ele cobre o mapa inteiro). O círculo vermelho é o caçador e o dourado, a moeda. Uma linha azul tracejada aparece quando o caçador vê a moeda. O painel mostra moedas, taxa de captura e se ele está usando visão ou olfato.

Para começar a janela da direita num mapa que o modelo nunca viu: `python v8\demo.py --direita grande_aleatorio`

## 3. Outros comandos

Todos rodam a partir da pasta `CoinChaseIA`.

```powershell
# uma janela só, num mapa escolhido
python v8\visualizar.py --mapa grande_girado

# taxa de captura do mapa atual, sem janela (~6 s para 30 episódios)
python v8\avaliar.py 30

# escolher o mapa por variável de ambiente (vale para avaliar.py, main.py e os testes)
$env:MAPA = "grande_aleatorio"; $env:SEMENTE = "7"
python v8\avaliar.py 30
Remove-Item Env:MAPA, Env:SEMENTE      # volta ao padrão (grande)
```

**Mapas:** `grande` (padrão), `grande_espelho`, `grande_vertical`, `grande_girado`, `grande_aleatorio` (cada `SEMENTE` gera um mapa diferente, sempre conexo) e `original` (20x20).

## 4. Treinar

```powershell
python v8\main.py                          # mapa grande
$env:MAPA = "original"; python v8\main.py  # mapa 20x20
```

- Retoma do último `checkpoint_latest.pth` da pasta do mapa e salva a cada 25 episódios (100 no 20x20). Para parar num total de episódios, contando os já treinados: `$env:EPISODIOS = "400"` (o modelo do 54x54 está no episódio 300, então isso treina mais 100).
- **Atenção:** o treino sobrescreve os modelos da pasta. Para não perder o modelo atual, copie a pasta antes.
- Se o mapa grande não tiver modelo, o treino começa dos pesos do modelo 20x20 (fine-tuning), e não do zero.
- Parâmetros (recompensas, exploração, passos por episódio) ficam em `v8/config.py`.

## 5. Os arquivos de modelo

Cada pasta de modelo tem dois tipos de arquivo:

| Arquivo | Tamanho | Para quê |
|---|---|---|
| `modelo.pth` | ~44 KB | Só os pesos da rede. **É o que a demo e os testes usam.** Abre com `torch.load(..., weights_only=True)`. |
| `checkpoint_latest.pth` (e `checkpoint_epN.pth`) | ~6,5 MB | Pesos + otimizador + memória de replay. Serve para **continuar o treino**. Precisa de numpy 2. |

O `main.py` atualiza o `modelo.pth` sozinho sempre que salva o `checkpoint_latest.pth`.

## 6. Como funciona (resumo)

- **Estado (13 números):** 5 de olfato (cheiro na casa do caçador e nas 4 vizinhas; o cheiro é `1/(d+1)`, com `d` = distância real até a moeda por busca em largura) e 8 de visão (moeda em linha reta ou diagonal, até 8 casas, sem parede no meio).
- **Rede:** 13 → 128 → 64 → 4 (ReLU), uma saída Q por ação (cima, baixo, esquerda, direita). Arquivo `v8/rede.py`.
- **Algoritmo:** DQN com replay de experiência (20 000 jogadas), rede-alvo e exploração ε-greedy. Arquivo `v8/agente.py`.
- **Recompensas:** +3 na captura, +0,2 quando o cheiro aumenta, penalidades por bater na parede e por andar em ciclos.
- Como o estado é local, o mesmo modelo funciona em mapas de qualquer tamanho.

## 7. Estrutura

```
CoinChaseIA/
├── demo_aula.bat          # dois cliques = demo lado a lado
├── requirements.txt
├── v5/ v6/ v7/            # versões anteriores (todas no 20x20)
└── v8/                    # versão atual
    ├── demo.py            # abre as duas janelas lado a lado
    ├── visualizar.py      # visualização com troca de mapa ao vivo
    ├── avaliar.py         # taxa de captura sem janela
    ├── main.py            # treino
    ├── config.py          # mapa e parâmetros
    ├── labirinto.py       # mapas 20x20, 54x54 e variações
    ├── moeda.py           # comportamento da moeda e cálculo do cheiro
    ├── agente.py          # agente DQN (estado, ações, aprendizado)
    ├── rede.py            # rede neural
    ├── teste_continuo.py / teste_continuo_grafico.py   # testes antigos (terminal / gráfico)
    ├── export_onix.py     # exporta o modelo para ONNX
    ├── models/            # modelos do 20x20
    └── models_grande/     # modelos do 54x54
```
