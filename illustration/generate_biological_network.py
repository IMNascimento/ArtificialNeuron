#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um GIF "rede_biologica.gif" com aparência realista de uma rede neural biológica
("pequeno cérebro"), fundo branco, neurônios em cores sólidas, e pulsos vermelho‑alaranjados
se propagando em ondas por ~20 segundos (320 frames a 16 fps) em 1920x1080.

Dependências: numpy, matplotlib (com pillow instalado para salvar GIF)
Execute:  python gerar_rede_biologica.py
Saída:    rede_biologica.gif (na mesma pasta)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# ======================== Parâmetros Gerais ========================
DURATION_S = 20
FPS = 16
N_FRAMES = DURATION_S * FPS

W, H = 1920, 1080
DPI = 100
FIGSIZE = (W / DPI, H / DPI)  # 1920x1080

# Rede
N_NEURONS = 28                 # quantidade de neurônios
SOMA_R_MIN, SOMA_R_MAX = 14, 26  # raio dos somas em pixels
CONN_K = 3                     # conexões por neurônio (aprox. vizinhos)
EDGE_WIDTH = 1.6               # espessura das conexões

# Pulsos / Firing
BASE_SPEED = 220.0            # px/seg velocidade base dos pulsos ao longo das conexões
WAVE_PERIOD = 6.5             # segundos entre "ondas" globais
WAVE_SEEDS = 3                # quantos focos de onda
SOMA_GLOW_FRAMES = int(0.35 * FPS)  # quanto tempo o soma "brilha" após disparar

# Cores (sólidas, fundo branco)
CLR_AXON = "#34495e"
CLR_DEND = "#3c6db3"
CLR_SOMA_EDGE = "#1c2833"
CLR_SOMA_FILL = "#a8d5c2"     # verde-azulado biológico
CLR_SOMA_IDLE = "#b7e1d4"     # variação para idle
CLR_PULSE_A = "#ff6f00"       # laranja forte
CLR_PULSE_B = "#d84315"       # vermelho-alaranjado
CLR_GLOW = "#ffcc80"          # halo amarelado durante spike

np.random.seed(7)

# ======================== Utilidades Geométricas ========================
def rand_in_ellipse(cx, cy, rx, ry):
    """Amostra ponto dentro de elipse via rejeição simples."""
    for _ in range(10000):
        x = np.random.uniform(cx - rx, cx + rx)
        y = np.random.uniform(cy - ry, cy + ry)
        if ((x - cx) ** 2) / (rx ** 2) + ((y - cy) ** 2) / (ry ** 2) <= 1.0:
            return x, y
    return cx, cy

def poisson_disk_like(n, rx, ry, min_dist):
    """Distribui ~n pontos dentro de elipse (0..W,0..H) evitando sobreposição excessiva."""
    pts = []
    cx, cy = W * 0.5, H * 0.5
    tries = 0
    while len(pts) < n and tries < n * 5000:
        tries += 1
        x, y = rand_in_ellipse(cx, cy, rx, ry)
        if not pts:
            pts.append((x, y))
            continue
        ok = True
        for px, py in pts:
            if (x - px) ** 2 + (y - py) ** 2 < (min_dist ** 2):
                ok = False
                break
        if ok:
            pts.append((x, y))
    return np.array(pts)

def quad_bezier(P0, P1, P2, t):
    """Bezier quadrática para t escalar ou array."""
    t = np.asarray(t)[..., None]
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

def curved_connection(a, b, curviness=0.25):
    """Cria uma curva suave entre pontos a e b usando Bezier quadrática com controle lateral."""
    a = np.asarray(a); b = np.asarray(b)
    mid = 0.5 * (a + b)
    vec = b - a
    # normal 2D
    n = np.array([-vec[1], vec[0]], dtype=float)
    n_norm = np.linalg.norm(n) + 1e-6
    n = n / n_norm
    ctrl = mid + n * curviness * np.linalg.norm(vec)
    t = np.linspace(0, 1, 160)
    curve = quad_bezier(a, ctrl, b, t)
    return curve  # (N,2)

def jitter_angle(theta, jitter=np.deg2rad(14)):
    return theta + np.random.uniform(-jitter, jitter)

# ======================== Geração da Rede ========================
# Colocamos neurônios em uma elipse central (pequeno "cérebro")
ellipse_rx = W * 0.38
ellipse_ry = H * 0.34
min_dist = 70  # distância mínima entre centros

centers = poisson_disk_like(N_NEURONS, ellipse_rx, ellipse_ry, min_dist)

# Raio dos somas
soma_r = np.random.uniform(SOMA_R_MIN, SOMA_R_MAX, size=N_NEURONS)

# Para desenhar "dendritos locais" (ornamentação orgânica ao redor do soma)
def dendrite_branches(center, R, count=6):
    """Gera pequenos ramos ao redor do soma para visual orgânico (não usados como conexões)."""
    cx, cy = center
    segments = []
    for _ in range(count):
        ang = np.random.uniform(0, 2 * np.pi)
        L = np.random.uniform(R * 0.9, R * 2.1)
        p0 = np.array([cx + np.cos(ang) * (R + 2), cy + np.sin(ang) * (R + 2)])
        p2 = np.array([cx + np.cos(ang) * (R + L), cy + np.sin(ang) * (R + L)])
        # controle com leve jitter
        ang_c = jitter_angle(ang, np.deg2rad(25))
        p1 = np.array([cx + np.cos(ang_c) * (R + L * 0.55), cy + np.sin(ang_c) * (R + L * 0.55)])
        t = np.linspace(0, 1, 45)
        curve = quad_bezier(p0, p1, p2, t)
        segs = np.stack([curve[:-1], curve[1:]], axis=1)
        segments.append(segs)
    if segments:
        return np.concatenate(segments, axis=0)
    return np.empty((0,2,2))

local_dend_segments = [dendrite_branches(centers[i], soma_r[i], count=np.random.randint(5, 9))
                       for i in range(N_NEURONS)]

# Conexões entre neurônios (grafo usando K vizinhos mais próximos)
def k_nearest_edges(centers, k):
    edges = set()
    for i, a in enumerate(centers):
        d = np.linalg.norm(centers - a, axis=1)
        idx = np.argsort(d)
        # ignorar o próprio (idx[0] == i), pegar próximos k+2 e filtrar para variedade
        for j in idx[1: k+3]:
            u, v = min(i, j), max(i, j)
            if u != v:
                edges.add((u, v))
    return sorted(edges)

edges = k_nearest_edges(centers, CONN_K)

# Para cada aresta, criamos uma curva suave (Bezier) e seu comprimento cumulativo para animação
edge_curves = []
edge_lengths = []
edge_cumlen = []
for (i, j) in edges:
    a = centers[i]; b = centers[j]
    # iniciar/terminar na borda dos somas (aprox.)
    vec = b - a
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        continue
    dir_ab = vec / dist
    A = a + dir_ab * (soma_r[i] + 4)  # 4 px gap
    B = b - dir_ab * (soma_r[j] + 4)
    curve = curved_connection(A, B, curviness=0.22 + np.random.uniform(-0.07, 0.07))
    # comprimento
    diff = curve[1:] - curve[:-1]
    seglen = np.sqrt((diff**2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    edge_curves.append(curve)
    edge_lengths.append(cum[-1])
    edge_cumlen.append(cum)

edge_curves = np.array(edge_curves, dtype=object)
edge_lengths = np.array(edge_lengths)
edge_cumlen = np.array(edge_cumlen, dtype=object)

# Mapa de adjacência para propagação de ondas (grafo não direcionado)
adj = {i: set() for i in range(N_NEURONS)}
for (i, j) in edges:
    adj[i].add(j); adj[j].add(i)

# ======================== Cronograma de "ondas" ========================
# Escolhemos alguns "seeds" que iniciam ondas periodicamente
seed_nodes = np.random.choice(np.arange(N_NEURONS), size=WAVE_SEEDS, replace=False)

# Para ativação de somas (glow), marcamos últimos frames em que cada neurônio disparou
last_fire_frame = np.full(N_NEURONS, -9999, dtype=int)

# Função que, dado um frame global, indica quais edges têm pulsos e
# em que posição ao longo da curva (0..len-1)
def generate_active_pulses(frame):
    """
    Gera pulsos a partir de seeds em ondas. Propagamos por BFS com atraso
    proporcional ao comprimento da aresta e velocidade BASE_SPEED.
    Retorna uma lista de (edge_index, pos_idx, is_alt_color, src_node, dst_node).
    """
    t_sec = frame / FPS
    pulses = []
    # Cada seed dispara a cada WAVE_PERIOD segundos (fase levemente diferente)
    for s_i, s in enumerate(seed_nodes):
        phase = (s_i / len(seed_nodes)) * (WAVE_PERIOD * 0.33)
        # número de ondas já iniciadas até o tempo atual
        n_waves = int((t_sec - phase) / WAVE_PERIOD) + 1
        for w in range(max(0, n_waves)):
            t0 = phase + w * WAVE_PERIOD  # início da onda
            if t_sec < t0:
                continue
            # BFS a partir do seed para calcular tempos de chegada aos nós
            # e disparos ao longo das arestas
            visited = {s: 0.0}  # tempo relativo em segundos
            queue = [s]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if v in visited:
                        continue
                    # Comprimento médio de uma conexão u-v (encontrar edge index)
                    # (pode haver uma única aresta u-v; procuramos)
                    length_uv = None
                    edge_idx = None
                    for idx, (i, j) in enumerate(edges):
                        if (i == u and j == v) or (i == v and j == u):
                            length_uv = edge_lengths[idx]
                            edge_idx = idx
                            break
                    if edge_idx is None or length_uv is None:
                        continue
                    # atraso para percorrer aresta
                    dt = length_uv / BASE_SPEED
                    visited[v] = visited[u] + dt
                    queue.append(v)
                    # No tempo atual t_sec, qual fração do caminho já foi percorrida?
                    # Considerando que o pulso partiu do seed no tempo t0
                    t_elapsed = t_sec - (t0 + visited[u])
                    if t_elapsed < 0:
                        continue
                    frac = t_elapsed / dt if dt > 1e-6 else 1.0
                    if 0.0 <= frac <= 1.0:
                        # posição ao longo da curva
                        cum = edge_cumlen[edge_idx]
                        L = edge_lengths[edge_idx]
                        s = frac * L
                        pos = np.searchsorted(cum, s)
                        pulses.append((edge_idx, pos, (w % 2 == 1), u, v))
                        # marcar os somas como "recentemente disparados"
                        # origem u e destino v em torno do momento de passagem
                        if abs(frac) < 0.04:
                            last_fire_frame[u] = frame
                        if abs(1.0 - frac) < 0.04:
                            last_fire_frame[v] = frame
    return pulses

# ======================== Setup de Desenho (Matplotlib) ========================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_facecolor("#f8f7f3")
ax.set_facecolor("#f8f7f3")

# Camada: dendritos locais (ornamentação)
dend_lc = []
for segs in local_dend_segments:
    lc = LineCollection(segs, colors=CLR_DEND, linewidths=0.9, alpha=0.85, zorder=2)
    ax.add_collection(lc)
    dend_lc.append(lc)

# Camada: conexões (axônios/sinapses)
conn_segs = []
for curve in edge_curves:
    segs = np.stack([curve[:-1], curve[1:]], axis=1)
    conn_segs.append(segs)
conn_lc = LineCollection(np.concatenate(conn_segs, axis=0) if conn_segs else np.empty((0,2,2)),
                         colors=CLR_AXON, linewidths=EDGE_WIDTH, alpha=0.9, zorder=3)
ax.add_collection(conn_lc)

# Camada: somas
soma_patches = []
for i in range(N_NEURONS):
    c = Circle(centers[i], soma_r[i], edgecolor=CLR_SOMA_EDGE, facecolor=CLR_SOMA_FILL, lw=1.4, zorder=5)
    ax.add_patch(c)
    soma_patches.append(c)

# Camada: halo do soma (glow durante spike)
halo_patches = []
for i in range(N_NEURONS):
    h = Circle(centers[i], soma_r[i] * 1.7, edgecolor="none", facecolor=CLR_GLOW, alpha=0.0, zorder=4)
    ax.add_patch(h)
    halo_patches.append(h)

# Camada: partículas (pulsos) — um "scatter" manual por edge para poucos marcadores
pulse_markers = []
MAX_PULSES = max(8, int(len(edges) * 0.35))  # limite de marcadores simultâneos por performance
for _ in range(MAX_PULSES):
    m, = ax.plot([], [], 'o', ms=6.0, mfc=CLR_PULSE_A, mec="none", alpha=0.0, zorder=6)
    pulse_markers.append(m)

# ======================== Animação ========================
def animate(frame):
    # Atualiza halos (glow) dos somas conforme "último disparo"
    for i in range(N_NEURONS):
        dt = frame - last_fire_frame[i]
        if 0 <= dt <= SOMA_GLOW_FRAMES:
            # opacidade decai ao longo do tempo
            alpha = max(0.0, 0.55 * (1.0 - dt / SOMA_GLOW_FRAMES))
            halo_patches[i].set_alpha(alpha)
            soma_patches[i].set_facecolor(CLR_SOMA_FILL)
        else:
            halo_patches[i].set_alpha(0.0)
            # somas oscilam discretamente de cor para aspecto "vivo"
            if (frame // 12 + i) % 2 == 0:
                soma_patches[i].set_facecolor(CLR_SOMA_FILL)
            else:
                soma_patches[i].set_facecolor(CLR_SOMA_IDLE)

    # Gera pulsos ativos neste frame
    pulses = generate_active_pulses(frame)

    # Desenha um número limitado de marcadores por frame para manter fluidez
    for k, marker in enumerate(pulse_markers):
        if k < len(pulses):
            edge_idx, pos_idx, alt, u, v = pulses[k]
            curve = edge_curves[edge_idx]
            pos_idx = np.clip(pos_idx, 0, len(curve)-1)
            x, y = curve[pos_idx]
            marker.set_data([x], [y])
            marker.set_alpha(1.0)
            marker.set_markersize(6.5 + 0.8 * math.sin(0.25 * frame + 0.9 * k))
            marker.set_color(CLR_PULSE_A if not alt else CLR_PULSE_B)
        else:
            marker.set_alpha(0.0)

    return []

if __name__ == "__main__":
    print("[*] Renderizando GIF: rede_biologica.gif ...")
    ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000/FPS, blit=False)
    ani.save("rede_biologica.gif", writer="pillow", fps=FPS, savefig_kwargs={'facecolor': '#f8f7f3'})
    print("[✓] Concluído: rede_biologica.gif")
