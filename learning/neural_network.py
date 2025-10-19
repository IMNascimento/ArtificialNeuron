from .neuron import Neuron
# ==========================================
# NEURAL NETWORK (hidden: sigmoid, output: linear)
# ==========================================

class NeuralNetwork:
    """
    Vários neurônios trabalhando em equipe
    Como um time: cada um faz sua parte, juntos resolvem problemas complexos!
    """

    def __init__(self, num_inputs, num_hidden, num_outputs,
                 hidden_activation="sigmoid", output_activation="linear"):
        # Build layers with the specified sizes and activations
        print(f"\n🏗️  Montando equipe de neurônios:")
        print(f"   • {num_inputs} entradas (recebem informação)")
        print(f"   • {num_hidden} neurônios pensadores (processam)")
        print(f"   • {num_outputs} neurônios decisores (respondem)")

        # Hidden layer (tipicamente sigmoid/tanh)
        self.hidden_layer = [
            Neuron(num_inputs, activation=hidden_activation) for _ in range(num_hidden)
        ]
        # Output layer (para regressão: linear; para classificação: sigmoid)
        self.output_layer = [
            Neuron(num_hidden, activation=output_activation) for _ in range(num_outputs)
        ]

    def think(self, inputs):
        """Propaga as entradas pela rede e retorna (saídas_finais, saídas_ocultas)"""
        hidden_outputs = [n.think(inputs) for n in self.hidden_layer]
        final_outputs  = [n.think(hidden_outputs) for n in self.output_layer]
        return final_outputs, hidden_outputs

    def train(self, training_data, epochs=20000, learning_rate=0.5):
        """Treina a rede ajustando os pesos com base no erro observado (backpropagation)"""
        print(f"\n📚 Treinando a rede com {len(training_data)} exemplos...")

        for epoch in range(epochs):
            for inputs, expected in training_data:
                # -------- Forward --------
                final_outputs, hidden_outputs = self.think(inputs)

                # -------- Output layer deltas --------
                # delta_out_j = (target_j - output_j) * f'_out(output_j)
                deltas_out = []
                for j, neuron in enumerate(self.output_layer):
                    error = expected[j] - final_outputs[j]
                    local_grad = neuron._activation_derivative(final_outputs[j])
                    delta = error * local_grad
                    deltas_out.append(delta)

                    # SGD update (output weights and bias)
                    for k in range(len(neuron.weights)):
                        neuron.weights[k] += hidden_outputs[k] * delta * learning_rate
                    neuron.bias += delta * learning_rate

                # -------- Hidden layer deltas --------
                # delta_h_i = (sum_j delta_out_j * w_j_i) * f'_hidden(hidden_outputs[i])
                for i, neuron in enumerate(self.hidden_layer):
                    back_error = 0.0
                    for j, out_neuron in enumerate(self.output_layer):
                        back_error += deltas_out[j] * out_neuron.weights[i]

                    local_grad_h = neuron._activation_derivative(hidden_outputs[i])
                    delta_h = back_error * local_grad_h

                    # SGD update (hidden weights and bias)
                    for k in range(len(neuron.weights)):
                        neuron.weights[k] += inputs[k] * delta_h * learning_rate
                    neuron.bias += delta_h * learning_rate

            # Progress report
            if epoch % 5000 == 0:
                total_error = 0.0
                for x, y in training_data:
                    y_pred, _ = self.think(x)
                    total_error += sum(abs(yt - yp) for yt, yp in zip(y, y_pred))
                print(f"  Treino {epoch}: Erro total = {total_error:.4f}")

        print("✅ Rede treinada!\n")