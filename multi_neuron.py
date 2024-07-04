import numpy as np

# Função para tokenizar as frases
def tokenize(sentence):
    return sentence.lower().split()

# Criar um vocabulário simples
def build_vocab(sentences):
    vocab = {}
    idx = 0
    for sentence in sentences:
        for word in tokenize(sentence):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

# Codificar frases usando o vocabulário
def encode_sentence(sentence, vocab):
    vector = np.zeros(len(vocab))
    for word in tokenize(sentence):
        if word in vocab:
            vector[vocab[word]] = 1
    return vector

# Dados de exemplo
sentences = [
    "oi tudo bem", "como você está", "olá", "oi meu amigo",
    "boa tarde", "bom dia", "oi e aí", "qual é o seu nome",
    "prazer em conhecer", "adeus", "oi como você está", "oi e bom dia",
    "olá tudo bem", "oi boa tarde", "como vai você", "tudo bem",
    "e aí tudo bem", "qual o seu nome", "oi bom dia", "oi oi"
]
labels = np.array([
    [1, 0], [0, 1], [0, 0], [1, 0],
    [0, 0], [0, 0], [1, 0], [0, 1],
    [0, 0], [0, 0], [1, 1], [1, 0],
    [0, 0], [1, 0], [0, 1], [0, 0],
    [0, 0], [0, 1], [1, 0], [1, 0]
])

# Construir o vocabulário e codificar as frases
vocab = build_vocab(sentences)
encoded_sentences = np.array([encode_sentence(sentence, vocab) for sentence in sentences])

# Dividir os dados em conjuntos de treinamento e validação
train_size = int(0.8 * len(encoded_sentences))
X_train, X_val = encoded_sentences[:train_size], encoded_sentences[train_size:]
y_train, y_val = labels[:train_size], labels[train_size:]

# Inicializar pesos e bias para duas camadas (entrada para oculta, oculta para saída)
np.random.seed(1)
input_size = len(vocab)  # Número de neurônios na camada de entrada
hidden_size = 10  # Número de neurônios na camada oculta
output_size = 2  # Dois neurônios na camada de saída: "oi" e "você"

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Função sigmoide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Taxa de aprendizado e número de épocas
learning_rate = 0.1
epochs = 10000

# Treinamento da rede neural
for epoch in range(epochs):
    # Propagação para frente
    Z1 = np.dot(X_train, W1) + b1  # Soma ponderada na camada oculta
    A1 = sigmoid(Z1)  # Ativação da camada oculta
    Z2 = np.dot(A1, W2) + b2  # Soma ponderada na camada de saída
    A2 = sigmoid(Z2)  # Ativação da camada de saída (saída final)

    # Calcular o erro
    error = y_train - A2

    # Backpropagation
    dZ2 = error * sigmoid_derivative(A2)  # Gradiente da camada de saída
    dW2 = np.dot(A1.T, dZ2)  # Gradiente dos pesos da camada de saída
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # Gradiente do bias da camada de saída
    
    dA1 = np.dot(dZ2, W2.T)  # Gradiente das ativações da camada oculta
    dZ1 = dA1 * sigmoid_derivative(A1)  # Gradiente da camada oculta
    dW1 = np.dot(X_train.T, dZ1)  # Gradiente dos pesos da camada oculta
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # Gradiente do bias da camada oculta

    # Atualização dos pesos e bias
    W1 += learning_rate * dW1
    b1 += learning_rate * db1
    W2 += learning_rate * dW2
    b2 += learning_rate * db2

    # Validação
    Z1_val = np.dot(X_val, W1) + b1  # Soma ponderada na camada oculta (validação)
    A1_val = sigmoid(Z1_val)  # Ativação da camada oculta (validação)
    Z2_val = np.dot(A1_val, W2) + b2  # Soma ponderada na camada de saída (validação)
    A2_val = sigmoid(Z2_val)  # Ativação da camada de saída (validação)
    val_loss = np.mean((y_val - A2_val) ** 2)

    if (epoch + 1) % 1000 == 0:
        train_loss = np.mean((y_train - A2) ** 2)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

# Função de predição
def predict(phrase, vocab, W1, b1, W2, b2):
    encoded_phrase = encode_sentence(phrase, vocab)
    Z1 = np.dot(encoded_phrase, W1) + b1  # Soma ponderada na camada oculta
    A1 = sigmoid(Z1)  # Ativação da camada oculta
    Z2 = np.dot(A1, W2) + b2  # Soma ponderada na camada de saída
    output = sigmoid(Z2)  # Ativação da camada de saída (saída final)
    return output.flatten()  # Garante que a saída seja um vetor 1D

# Testar o modelo com novas frases
test_phrases = ["oi você", "bom dia para todos", "oi como vai", "até logo", "oi meu caro"]
for phrase in test_phrases:
    prediction = predict(phrase, vocab, W1, b1, W2, b2)
    print(f"Frase: '{phrase}' - Predição: {prediction}")

# Salvar os pesos e bias treinados para uso posterior
np.save('W1.npy', W1)
np.save('b1.npy', b1)
np.save('W2.npy', W2)
np.save('b2.npy', b2)

# Carregar os pesos e bias treinados
W1 = np.load('W1.npy')
b1 = np.load('b1.npy')
W2 = np.load('W2.npy')
b2 = np.load('b2.npy')

# Loop contínuo para entrada do usuário
while True:
    user_input = input("Digite uma frase (ou 'sair' para terminar): ")
    if user_input.lower() == 'sair':
        break
    prediction = predict(user_input, vocab, W1, b1, W2, b2)
    print(f'Predição: "oi": {"Sim" if prediction[0] >= 0.5 else "Não"}, "você": {"Sim" if prediction[1] >= 0.5 else "Não"} - Valores: {prediction}')