#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um GIF "rede_hibrida.gif" misturando visual biológico (neurônios orgânicos) com layout de camadas.
Fundo branco; pulsos verde-azulados entre camadas; 20 s (320 frames @ 16 fps), 1920x1080.

Dependências: numpy, matplotlib (com pillow)
Execute:  python gerar_rede_hibrida.py
Saída:    rede_hibrida.gif
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# ------------------- Parâmetros -------------------
DURATION_S = 20
FPS = 16
N_FRAMES = DURATION_S * FPS

W, H = 1920, 1080
DPI = 100
FIGSIZE = (W / DPI, H / DPI)

LAYERS = [6, 8, 5]  # entrada, oculta, saída
LAYER_X = np.linspace(260, W-260, len(LAYERS))

SOMA_R = 18
EDGE_W = 2.2

CLR_BG = "#f8f7f3"
CLR_AXON = "#2c3e50"
CLR_DEND = "#3c6db3"
CLR_SOMA_EDGE = "#1c2833"
CLR_SOMA_FILL = "#a8d5c2"     # verde-azulado (biológico)
CLR_SOMA_IDLE = "#b7e1d4"     # variação
CLR_PULSE_A = "#00c853"       # verde (artificial)
CLR_PULSE_B = "#2196f3"       # azul (artificial)
CLR_GLOW = "#a5d6a7"          # halo suave

np.random.seed(12)

# ------------------- Layout dos neurônios (em camadas) -------------------
def layer_y_positions(n, top=160, bottom=H-160):
    return np.linspace(top, bottom, n)

centers = []
for li, n in enumerate(LAYERS):
    xs = np.full(n, LAYER_X[li])
    ys = layer_y_positions(n)
    # jitter orgânico na posição (pequena variação)
    xs = xs + np.random.uniform(-30, 30, size=n)
    ys = ys + np.random.uniform(-22, 22, size=n)
    centers.append(np.stack([xs, ys], axis=1))
centers = [np.array(layer) for layer in centers]

# Gerar "dendritos locais" (ornamentação) com curvas Bezier ao redor de cada soma
def quad_bezier(P0, P1, P2, t):
    t = np.asarray(t)[..., None]
    return (1-t)**2 * P0 + 2*(1-t)*t*P1 + t**2 * P2

def dendrite_branches(center, count=6, R=18):
    cx, cy = center
    segments = []
    for _ in range(count):
        ang = np.random.uniform(0, 2*np.pi)
        L = np.random.uniform(R*0.9, R*2.0)
        p0 = np.array([cx + np.cos(ang) * (R + 2), cy + np.sin(ang) * (R + 2)])
        p2 = np.array([cx + np.cos(ang) * (R + L), cy + np.sin(ang) * (R + L)])
        ang_c = ang + np.deg2rad(np.random.uniform(-25, 25))
        p1 = np.array([cx + np.cos(ang_c) * (R + L*0.6), cy + np.sin(ang_c) * (R + L*0.6)])
        t = np.linspace(0, 1, 44)
        curve = quad_bezier(p0, p1, p2, t)
        segs = np.stack([curve[:-1], curve[1:]], axis=1)
        segments.append(segs)
    return np.concatenate(segments, axis=0) if segments else np.empty((0,2,2))

local_dend_segments = []
for layer in centers:
    for c in layer:
        local_dend_segments.append(dendrite_branches(c, count=np.random.randint(5, 8), R=SOMA_R))

# Conexões entre camadas (de todos-para-todos entre camadas consecutivas)
edges = []
for li in range(len(LAYERS)-1):
    a_layer = centers[li]
    b_layer = centers[li+1]
    for i, a in enumerate(a_layer):
        for j, b in enumerate(b_layer):
            edges.append((li, i, li+1, j))

# Curvas levemente orgânicas para conexões
def curved_connection(a, b, curviness=0.16):
    a = np.asarray(a); b = np.asarray(b)
    mid = 0.5*(a+b)
    vec = b-a
    n = np.array([-vec[1], vec[0]], dtype=float)
    n /= (np.linalg.norm(n) + 1e-6)
    ctrl = mid + n * curviness * np.linalg.norm(vec)
    t = np.linspace(0, 1, 120)
    return quad_bezier(a, ctrl, b, t)

edge_curves = []
edge_cumlen = []
edge_lengths = []
for (l1, i, l2, j) in edges:
    a = centers[l1][i]
    b = centers[l2][j]
    # deslocar início/fim para borda do soma (aprox)
    vec = b - a
    dist = np.linalg.norm(vec) + 1e-6
    dir_ab = vec / dist
    A = a + dir_ab * (SOMA_R + 3)
    B = b - dir_ab * (SOMA_R + 3)
    curve = curved_connection(A, B, curviness=0.14 + np.random.uniform(-0.05, 0.05))
    edge_curves.append(curve)
    diff = curve[1:] - curve[:-1]
    seglen = np.sqrt((diff**2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    edge_cumlen.append(cum)
    edge_lengths.append(cum[-1])

edge_curves = np.array(edge_curves, dtype=object)
edge_cumlen = np.array(edge_cumlen, dtype=object)
edge_lengths = np.array(edge_lengths)

# Índices por camada de origem
edges_by_layer = {}
for idx, (l1, i, l2, j) in enumerate(edges):
    edges_by_layer.setdefault(l1, []).append(idx)

# ------------------- Desenho -------------------
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
fig.patch.set_facecolor(CLR_BG); ax.set_facecolor(CLR_BG)

# Dendritos locais
for segs in local_dend_segments:
    lc = LineCollection(segs, colors=CLR_DEND, linewidths=1.0, alpha=0.85, zorder=2)
    ax.add_collection(lc)

# Conexões entre camadas
for curve in edge_curves:
    segs = np.stack([curve[:-1], curve[1:]], axis=1)
    lc = LineCollection(segs, colors=CLR_AXON, linewidths=EDGE_W, alpha=0.9, zorder=3)
    ax.add_collection(lc)

# Somas (neurônios biológicos em camadas)
soma_patches = []
for layer in centers:
    for (x, y) in layer:
        c = Circle((x, y), SOMA_R, ec=CLR_SOMA_EDGE, fc=CLR_SOMA_FILL, lw=1.4, zorder=5)
        soma_patches.append(c); ax.add_patch(c)

# Halos (glow)
halo_patches = []
for layer in centers:
    for (x, y) in layer:
        h = Circle((x, y), SOMA_R*1.6, ec="none", fc=CLR_GLOW, alpha=0.0, zorder=4)
        halo_patches.append(h); ax.add_patch(h)

# Pulsos
MAX_PULSES = 80
pulse_artists = []
for k in range(MAX_PULSES):
    m, = ax.plot([], [], 'o', ms=6.5, mfc=CLR_PULSE_A if k%2==0 else CLR_PULSE_B,
                 mec="none", alpha=0.0, zorder=6)
    pulse_artists.append(m)

# ------------------- Animação -------------------
def activation_waves(frame):
    # ondas varrendo camadas, 3 ondas defasadas
    t = frame / FPS
    phases = [0.0, 0.8, 1.6]
    speed = 1.0/6.5
    return [ (t*speed + p) % 1.0 for p in phases ]

def animate(frame):
    waves = activation_waves(frame)

    # brilho nos somas com leve alternância (vivo)
    for idx, c in enumerate(soma_patches):
        if (frame // 10 + idx) % 2 == 0:
            c.set_facecolor(CLR_SOMA_FILL)
        else:
            c.set_facecolor(CLR_SOMA_IDLE)
        # halos respirando suave
        h = halo_patches[idx]
        alpha = 0.10 + 0.06*np.sin(0.2*frame + 0.7*idx)
        h.set_alpha(alpha)

    # animar pulsos ao longo das conexões por camada
    k = 0
    for li in range(len(LAYERS)-1):
        edx = edges_by_layer.get(li, [])
        if not edx: continue
        npl = min(16, len(edx)//3 + 3)
        pos = max(waves)  # 0..1
        for p in range(npl):
            if k >= len(pulse_artists): break
            idx = edx[(frame*3 + p*11) % len(edx)]
            curve = edge_curves[idx]
            cum = edge_cumlen[idx]; L = edge_lengths[idx]
            s = pos * L
            pos_idx = np.searchsorted(cum, s)
            pos_idx = np.clip(pos_idx, 0, len(curve)-1)
            x, y = curve[pos_idx]
            pulse_artists[k].set_data([x],[y])
            pulse_artists[k].set_alpha(1.0)
            pulse_artists[k].set_markersize(6.2 + 0.8*math.sin(0.33*frame + 0.8*k))
            k += 1
    while k < len(pulse_artists):
        pulse_artists[k].set_alpha(0.0)
        k += 1
    return []

if __name__ == "__main__":
    print("[*] Renderizando GIF: rede_hibrida.gif ...")
    ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000/FPS, blit=False)
    ani.save("rede_hibrida.gif", writer="pillow", fps=FPS, savefig_kwargs={'facecolor': CLR_BG})
    print("[✓] Concluído: rede_hibrida.gif")
