import random
import math

def tanh(x):
    if x > 20: return 1.0
    if x < -20: return -1.0
    e2x = math.exp(2*x)
    return (e2x - 1) / (e2x + 1)

def dtanh(y):  # y já ativado
    return 1.0 - y*y

ACTS = {
    "tanh": (tanh, dtanh),
    "linear": (lambda x: x, lambda _y: 1.0),
}

def sgn(x):
    return 0.0 if x == 0 else (1.0 if x > 0 else -1.0)


class MLP:
    """
    sizes: [n_in, n_hidden, n_out], ex.: [2, 16, 1]
    hidden_activation: 'tanh'
    output_activation: 'linear'
    """
    def __init__(self, sizes, hidden_activation="tanh", output_activation="linear", seed=42):
        random.seed(seed)
        self.sizes = sizes
        self.hidden_act = ACTS[hidden_activation]
        self.output_act = ACTS[output_activation]

        self.W = []
        self.b = []
        for l in range(len(sizes)-1):
            fan_in, fan_out = sizes[l], sizes[l+1]
            limit = math.sqrt(6.0/(fan_in+fan_out))  # Xavier
            self.W.append([[random.uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)])
            self.b.append([0.0 for _ in range(fan_out)])

    def _forward(self, x):
        a = x[:]
        caches = []  # (z, a)
        for l in range(len(self.W)):
            z = []
            for j in range(len(self.W[l])):
                s = sum(self.W[l][j][i]*a[i] for i in range(len(a))) + self.b[l][j]
                z.append(s)
            act, _ = self.hidden_act if l < len(self.W)-1 else self.output_act
            a = [act(v) for v in z]
            caches.append((z, a))
        return caches

    def predict(self, x):
        return self._forward(x)[-1][1]

    def train(self, data, epochs=800, lr=0.02, clip=5.0, verbose_every=100):
        L = len(self.W)
        for epoch in range(1, epochs+1):
            random.shuffle(data)
            total = 0.0
            for x, y in data:
                caches = self._forward(x)
                y_pred = caches[-1][1][0]
                err = y_pred - y
                total += 0.5*err*err

                # backprop
                gW = [ [ [0.0]*len(self.W[l][0]) for _ in range(len(self.W[l])) ] for l in range(L) ]
                gb = [ [0.0]*len(self.b[l]) for l in range(L) ]
                da = [err]  # dL/d(a_L)

                for l in reversed(range(L)):
                    _z, a = caches[l]
                    _act, dact = self.hidden_act if l < L-1 else self.output_act
                    dz = [da[j]*dact(a[j]) for j in range(len(a))]
                    a_prev = x if l == 0 else caches[l-1][1]

                    for j in range(len(self.W[l])):
                        for i in range(len(self.W[l][j])):
                            gW[l][j][i] = dz[j]*a_prev[i]
                        gb[l][j] = dz[j]

                    da_prev = [0.0]*len(a_prev)
                    for i in range(len(a_prev)):
                        da_prev[i] = sum(self.W[l][j][i]*dz[j] for j in range(len(self.W[l])))
                    da = da_prev

                # clipping
                if clip is not None:
                    def clipv(v): return max(-clip, min(clip, v))
                    for l in range(L):
                        gb[l] = [clipv(v) for v in gb[l]]
                        for j in range(len(gW[l])):
                            gW[l][j] = [clipv(v) for v in gW[l][j]]

                # SGD
                for l in range(L):
                    for j in range(len(self.W[l])):
                        for i in range(len(self.W[l][j])):
                            self.W[l][j][i] -= lr*gW[l][j][i]
                        self.b[l][j] -= lr*gb[l][j]

            if verbose_every and epoch % verbose_every == 0:
                print(f"[época {epoch}] loss médio: {total/len(data):.6f}")


class MultiplicationNet:
    """
    Aprende multiplicação via: log|y| = log|x| + log|z|
    - Entrada da MLP: [log|x|, log|z|, sgn(x), sgn(z), is_zero]
      (is_zero ajuda a rede a ignorar alvos de log quando há zero)
    - Saída da MLP: valor escalar que representa log|y|
    - Sinal e zeros são tratados fora da rede (regras exatas).
    """
    def __init__(self, hidden=16, seed=123, eps=1e-12):
        self.eps = eps
        self.net = MLP([5, hidden, 1], hidden_activation="tanh", output_activation="linear", seed=seed)

    def _encode_input(self, x, z):
        is_zero = 1.0 if (x == 0.0 or z == 0.0) else 0.0
        lx = 0.0 if x == 0.0 else math.log(abs(x) + self.eps)
        lz = 0.0 if z == 0.0 else math.log(abs(z) + self.eps)
        sx = sgn(x)
        sz = sgn(z)
        return [lx, lz, sx, sz, is_zero]

    def _target_log_abs(self, x, z):
        if x == 0.0 or z == 0.0:
            # alvo ignorado quando is_zero=1; devolvemos 0 por convenção
            return 0.0
        return math.log(abs(x*z) + self.eps)

    def fit(self, n_samples=20000, mag_range=(1e-3, 1e3), exclude_band=None,
            epochs=800, lr=0.02, verbose_every=100):
        """
        Gera pares (x,z) aleatórios cobrindo várias escalas e sinais.
        - mag_range: faixa de magnitude |x|, |z|
        - exclude_band: (a,b) exclui magnitudes em [a,b] para simular 'não viu'
        """
        lo, hi = mag_range
        data = []

        def sample_mag():
            # Amostragem log-uniforme para cobrir muitos ordens de grandeza
            u = random.random()
            return math.exp(math.log(lo) + u*(math.log(hi) - math.log(lo)))

        for _ in range(n_samples):
            # decide zeros raramente para ensinar o caso especial
            if random.random() < 0.02:
                x = 0.0 if random.random() < 0.5 else (sample_mag()*random.choice([-1,1]))
                z = 0.0
            else:
                # magnitude + sinal
                def draw_one():
                    while True:
                        m = sample_mag()
                        if exclude_band is not None and (exclude_band[0] <= m <= exclude_band[1]):
                            # pula a banda excluída
                            continue
                        return m
                mx = draw_one()
                mz = draw_one()
                x = mx * random.choice([-1, 1])
                z = mz * random.choice([-1, 1])

            x_in = self._encode_input(x, z)
            y_logabs = self._target_log_abs(x, z)
            data.append((x_in, y_logabs))

        self.net.train(data, epochs=epochs, lr=lr, verbose_every=verbose_every)
        return self

    def predict(self, x, z):
        """Retorna y ≈ x*z (com regra exata para zero/sinal)."""
        if x == 0.0 or z == 0.0:
            return 0.0
        x_in = self._encode_input(x, z)
        y_logabs = self.net.predict(x_in)[0]
        sign = sgn(x)*sgn(z)
        y = sign * math.exp(y_logabs)
        return y


