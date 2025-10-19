from typing import List, Dict, Tuple
import math
import random

from .neuron import Neuron

def softmax(logits: List[float]) -> List[float]:
    """Softmax estável numericamente"""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

class GenerativeNetwork:
    """
    Rede que GERA texto novo!
    Como? Aprende padrões de sequências e cria continuações
    """

    def __init__(self, vocab_size: int, num_hidden: int):
        print(f"\n🤖 Criando IA Generativa:")
        print(f"   • Vocabulário: {vocab_size} caracteres diferentes")
        print(f"   • Neurônios pensadores: {num_hidden}")

        # Camada oculta: sigmoid
        self.hidden_layer: List[Neuron] = [
            Neuron(vocab_size, activation="sigmoid") for _ in range(num_hidden)
        ]
        # Camada de saída: logits lineares (softmax aplicado externamente)
        self.output_layer: List[Neuron] = [
            Neuron(num_hidden, activation="linear") for _ in range(vocab_size)
        ]

        self.vocab_size = vocab_size
        self.num_hidden = num_hidden

    # ---------- Forward helpers ----------
    def _forward_hidden(self, inputs: List[float]) -> List[float]:
        # entradas podem ser one-hot ou densas (tamanho = vocab_size)
        return [n.think(inputs) for n in self.hidden_layer]

    def _forward_output_logits(self, hidden_outputs: List[float]) -> List[float]:
        # retorna logits lineares dos neurônios de saída
        return [n.think(hidden_outputs) for n in self.output_layer]

    def think(self, inputs: List[float]) -> Tuple[List[float], List[float]]:
        """Processa entrada e prevê próximo caractere (retorna probabilidades e saídas ocultas)"""
        hidden_outputs = self._forward_hidden(inputs)
        logits = self._forward_output_logits(hidden_outputs)
        probs = softmax(logits)
        return probs, hidden_outputs

    # ---------- Training (next-char prediction) ----------
    def train(self,
              train_data: List[Tuple[List[float], List[float]]],
              epochs: int = 5000,
              learning_rate: float = 0.1,
              report_every: int = 500):
        """
        📚 Treinando IA com exemplos de padrões (prever próximo caractere)
        Cada item de train_data é (x_onehot, y_onehot), ambos do tamanho do vocabulário.
        """
        print(f"\n📚 Treinando IA com {len(train_data)} exemplos de padrões...")

        for epoch in range(1, epochs + 1):
            random.shuffle(train_data)
            epoch_loss = 0.0

            for x_onehot, y_onehot in train_data:
                # ----- forward -----
                hidden = self._forward_hidden(x_onehot)
                logits = self._forward_output_logits(hidden)
                probs = softmax(logits)

                # Entropia cruzada: -log(probabilidade do alvo)
                try:
                    target_idx = y_onehot.index(1)
                except ValueError:
                    # fallback se vier float: usa índice do maior valor
                    target_idx = max(range(len(y_onehot)), key=lambda i: y_onehot[i])

                p_target = max(1e-12, probs[target_idx])
                loss = -math.log(p_target)
                epoch_loss += loss

                # Gradiente clássico de softmax + cross-entropy: dL/dlogit = probs - y_onehot
                deltas_out = [probs[i] - y_onehot[i] for i in range(self.vocab_size)]

                # Atualiza camada de saída (SGD)
                for i, neuron in enumerate(self.output_layer):
                    delta = deltas_out[i]
                    # w_k := w_k - lr * (hidden[k] * delta)
                    for k in range(len(neuron.weights)):
                        neuron.weights[k] -= learning_rate * (hidden[k] * delta)
                    neuron.bias -= learning_rate * delta

                # Backprop para a camada oculta
                # erro_oculto_j = soma_i( delta_out_i * W_out_i_j )
                for j, neuron_h in enumerate(self.hidden_layer):
                    back_error = 0.0
                    for i, out_neuron in enumerate(self.output_layer):
                        back_error += deltas_out[i] * out_neuron.weights[j]
                    # derivada da sigmoid em termo da saída já ativada
                    local_grad_h = neuron_h._activation_derivative(hidden[j])
                    delta_h = back_error * local_grad_h

                    # w_in := w_in - lr * (x * delta_h)
                    for k in range(len(neuron_h.weights)):
                        neuron_h.weights[k] -= learning_rate * (x_onehot[k] * delta_h)
                    neuron_h.bias -= learning_rate * delta_h

            if epoch % report_every == 0 or epoch == 1:
                print(f"  Época {epoch}: Loss médio = {epoch_loss / max(1, len(train_data)):.4f}")

        print("✅ IA treinada e pronta para gerar!\n")

    # ---------- Generation ----------
    def predict_next(self, x_onehot: List[float]) -> List[float]:
        """
        GERAÇÃO: Prevê qual caractere vem depois
        Retorna probabilidades para cada possível caractere
        """
        probs, _ = self.think(x_onehot)
        return probs

    def generate_text(self,
                      start_text: str,
                      vocab: List[str],
                      char_to_idx: Dict[str, int],
                      idx_to_char: Dict[int, str],
                      length: int = 50) -> str:
        """
        GERA TEXTO letra por letra!
        Como ChatGPT: vê o que já tem, prevê o próximo, adiciona, repete...
        (greedy: escolhe o de maior probabilidade)
        """
        generated = start_text

        for _ in range(length):
            if not generated:
                break
            last_char = generated[-1]
            if last_char not in char_to_idx:
                break

            # monta one-hot da última letra
            x = [0.0] * self.vocab_size
            x[char_to_idx[last_char]] = 1.0

            # prevê próximo e escolhe o mais provável
            probs = self.predict_next(x)
            next_idx = max(range(self.vocab_size), key=lambda i: probs[i])
            next_char = idx_to_char[next_idx]
            generated += next_char

        return generated
