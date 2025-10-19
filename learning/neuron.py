import random
import math

# ==========================================
# ARTIFICIAL NEURON CLASS
# ==========================================

class Neuron:
    """
    Um neurônio artificial que imita o cérebro humano:
    - Recebe informações (entradas)
    - Processa com seus "pesos" (memória/conhecimento)
    - Decide uma resposta (saída)
    """
    
    def __init__(self, num_inputs, activation="linear"):
        # Random init (keep small for stability with linear output)
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.activation = activation  # "linear" or "sigmoid"
        
    def _sigmoid(self, x):
        """Função que "suaviza" a resposta entre 0 e 1"""
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def _activate(self, x):
        # Choose activation
        if self.activation == "sigmoid":
            return self._sigmoid(x)
        # Linear (identity) for regression
        return x
    
    def _activation_derivative(self, y):
        # y here is already activation(x)
        if self.activation == "sigmoid":
            return y * (1 - y)
        # Derivative of identity is 1
        return 1.0
    
    def think(self, inputs):
        """O neurônio processa as entradas e dá uma resposta"""
        total = sum(i * w for i, w in zip(inputs, self.weights)) + self.bias
        return self._activate(total)
    
    def train(self, training_data, epochs=10000, learning_rate=0.1):
        """
        Ensina o neurônio através de exemplos repetidos
        Como uma criança aprendendo: tenta, erra, corrige, aprende!
        """
        print(f"📚 Ensinando o neurônio com {len(training_data)} exemplos...")
        print(f"   Vai praticar {epochs} vezes!\n")
        
        for epoch in range(epochs):
            for inputs, target in training_data:
                # Forward
                output = self.think(inputs)
                # Error (MSE gradient step)
                error = target - output
                # Local gradient
                local_grad = self._activation_derivative(output)
                adjustment = error * local_grad * learning_rate
                # Weights update
                for i in range(len(self.weights)):
                    self.weights[i] += inputs[i] * adjustment
                self.bias += adjustment

            # Progress
            if epoch % 2000 == 0:
                avg_error = sum(abs(t - self.think(x)) for x, t in training_data) / len(training_data)
                print(f"  Tentativa {epoch}: Erro médio = {avg_error:.4f}")
        
        print("✅ Aprendeu!\n")
