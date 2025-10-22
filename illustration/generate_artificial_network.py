#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um GIF "rede_computacional.gif" com estilo geométrico/limpo de uma rede neural artificial
(entrada -> oculta -> saída), fundo branco, pulsos verde/azulados propagando da esquerda para a direita.
Duração ~20 s (320 frames @ 16 fps), 1920x1080.

Dependências: numpy, matplotlib (com pillow)
Execute:  python gerar_rede_computacional.py
Saída:    rede_computacional.gif
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------- Parâmetros -------------------
DURATION_S = 20
FPS = 16
N_FRAMES = DURATION_S * FPS

W, H = 1920, 1080
DPI = 100
FIGSIZE = (W / DPI, H / DPI)

LAYERS = [6, 9, 6, 3]  # 4 camadas: entrada, 2 ocultas, saída
LAYER_X = np.linspace(260, W-260, len(LAYERS))

NODE_R = 20
EDGE_W = 2.0

CLR_BG = "#f8f7f3"
CLR_NODE_EDGE = "#1c2833"
CLR_NODE_FILL = "#f7f7f7"
CLR_EDGE = "#c3cbd3"         # linhas cinza suave
CLR_PULSE_A = "#00c853"      # verde
CLR_PULSE_B = "#2196f3"      # azul
CLR_HEAT_MIN = "#e8f5e9"     # preenchimento de ativação fraca (verde muito claro)
CLR_HEAT_MAX = "#a5d6a7"     # ativação forte (verde médio)

np.random.seed(10)

# ------------------- Layout dos nós -------------------
def layer_y_positions(n, top=140, bottom=H-140):
    ys = np.linspace(top, bottom, n)
    return ys

nodes = []  # lista de (x,y)
for li, n in enumerate(LAYERS):
    xs = np.full(n, LAYER_X[li])
    ys = layer_y_positions(n)
    nodes.append(np.stack([xs, ys], axis=1))
nodes = [np.array(layer) for layer in nodes]

# Conexões: totalmente conectadas entre camadas consecutivas
edges = []
for li in range(len(LAYERS)-1):
    for i, a in enumerate(nodes[li]):
        for j, b in enumerate(nodes[li+1]):
            edges.append((li, i, li+1, j))  # (layer_i, idx_i, layer_j, idx_j)

# Pré-calcular segmentos das arestas
edge_segments = []
for (l1, i, l2, j) in edges:
    a = nodes[l1][i]
    b = nodes[l2][j]
    seg = np.stack([a, b])
    edge_segments.append(seg)
edge_segments = np.array(edge_segments)

# ------------------- Ativação (propagação) -------------------
# Criamos uma "esteira" que move da esquerda -> direita repetidamente
def activation_schedule(frame):
    # fase normalizada 0..1 ao longo da largura; percorre camadas
    # 1 ciclo ~ 6s (96 frames). Usamos várias ondas sobrepostas.
    t = frame / FPS
    # quatro ondas defasadas para dar riqueza visual
    phases = [0.0, 1.2, 2.4, 3.6]
    speed = 1.0/6.0  # ciclos por segundo
    waves = []
    for p in phases:
        waves.append((t*speed + p) % 1.0)
    return waves  # lista de fases

def node_activation_strength(x, waves):
    # intensidade de 0..1 em função da proximidade da onda ao x da camada
    # mapeamos posição X dos layers para 0..1
    x_min, x_max = LAYER_X[0], LAYER_X[-1]
    x_norm = (x - x_min) / (x_max - x_min + 1e-6)
    val = 0.0
    for w in waves:
        d = min(abs(x_norm - w), 1.0 - abs(x_norm - w))  # distância com wrap
        contrib = max(0.0, 1.0 - (d / 0.08))            # largura da onda
        val = max(val, contrib)
    return val

# ------------------- Desenho -------------------
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
fig.patch.set_facecolor(CLR_BG); ax.set_facecolor(CLR_BG)

# Linhas (arestas)
for seg in edge_segments:
    ax.plot(seg[:,0], seg[:,1], '-', lw=EDGE_W, color=CLR_EDGE, alpha=0.85, zorder=1)

# Nós
node_artists = []
for li, layer in enumerate(nodes):
    for (x, y) in layer:
        c = plt.Circle((x, y), NODE_R, ec=CLR_NODE_EDGE, fc=CLR_NODE_FILL, lw=1.4, zorder=3)
        ax.add_patch(c)
        node_artists.append(c)

# Pulsos (marcadores se movendo pelas arestas)
MAX_PULSES = 90
pulse_artists = []
for k in range(MAX_PULSES):
    m, = ax.plot([], [], 'o', ms=6.5, mfc=CLR_PULSE_A if k%2==0 else CLR_PULSE_B,
                 mec="none", alpha=0.0, zorder=4)
    pulse_artists.append(m)

# Pré-índices das arestas por camada origem para animar mais fácil
edges_by_layer = {}
for idx, (l1, i, l2, j) in enumerate(edges):
    edges_by_layer.setdefault(l1, []).append(idx)

def animate(frame):
    waves = activation_schedule(frame)

    # Atualiza preenchimento dos nós conforme intensidade (gradiente simples entre duas cores)
    for li, layer in enumerate(nodes):
        for nj, (x, y) in enumerate(layer):
            strength = node_activation_strength(x, waves)
            # interpola entre CLR_NODE_FILL e CLR_HEAT_MAX
            # simples blending em RGB
            def hex_to_rgb(h): 
                h=h.lstrip('#'); return tuple(int(h[i:i+2],16) for i in (0,2,4))
            def rgb_to_hex(rgb): 
                return '#%02x%02x%02x' % rgb
            a = np.array(hex_to_rgb(CLR_NODE_FILL), dtype=float)
            b = np.array(hex_to_rgb(CLR_HEAT_MAX), dtype=float)
            rgb = (a*(1-strength) + b*strength).astype(int)
            node_artists[sum(LAYERS[:li]) + nj].set_facecolor(rgb_to_hex(tuple(rgb)))

    # Desenha pulsos ao longo de subconjuntos de arestas dependendo da onda
    # Selecionamos algumas arestas por camada, com posição ao longo [0..1]
    k = 0
    for li in range(len(LAYERS)-1):
        edx = edges_by_layer.get(li, [])
        if not edx: 
            continue
        # número de pulsos por camada/frame
        npl = min(18, len(edx)//3 + 4)
        # posição ao longo da aresta determinada por waves (pega a maior)
        pos = max(waves)
        for p in range(npl):
            if k >= len(pulse_artists): break
            idx = edx[(frame*3 + p*7) % len(edx)]
            (l1, i, l2, j) = edges[idx]
            a = nodes[l1][i]; b = nodes[l2][j]
            P = a*(1-pos) + b*pos
            pulse_artists[k].set_data([P[0]], [P[1]])
            pulse_artists[k].set_alpha(1.0)
            pulse_artists[k].set_markersize(6.2 + 0.9*math.sin(0.35*frame + 0.7*k))
            k += 1
    # Desativa pulsos restantes
    while k < len(pulse_artists):
        pulse_artists[k].set_alpha(0.0)
        k += 1
    return []

if __name__ == "__main__":
    print("[*] Renderizando GIF: rede_computacional.gif ...")
    ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000/FPS, blit=False)
    ani.save("rede_computacional.gif", writer="pillow", fps=FPS, savefig_kwargs={'facecolor': CLR_BG})
    print("[✓] Concluído: rede_computacional.gif")
