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
labels = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1])

# Construir o vocabulário e codificar as frases
vocab = build_vocab(sentences)
encoded_sentences = np.array([encode_sentence(sentence, vocab) for sentence in sentences])

# Dividir os dados em conjuntos de treinamento e validação
train_size = int(0.8 * len(encoded_sentences))
X_train, X_val = encoded_sentences[:train_size], encoded_sentences[train_size:]
y_train, y_val = labels[:train_size], labels[train_size:]

# Inicializar pesos e bias
np.random.seed(1)
weights = np.random.rand(len(vocab))
bias = np.random.rand(1)

# Função sigmoide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Taxa de aprendizado e número de épocas
learning_rate = 0.1
epochs = 10000

# Treinamento do perceptron com monitoramento da perda de validação
for epoch in range(epochs):
    # Treinamento
    inputs = X_train
    weighted_sum = np.dot(inputs, weights) + bias
    activated_output = sigmoid(weighted_sum)
    error = y_train - activated_output
    adjustments = error * sigmoid_derivative(activated_output)
    weights += np.dot(inputs.T, adjustments) * learning_rate
    bias += np.sum(adjustments) * learning_rate

    # Validação
    val_inputs = X_val
    val_weighted_sum = np.dot(val_inputs, weights) + bias
    val_activated_output = sigmoid(val_weighted_sum)
    val_loss = np.mean((y_val - val_activated_output) ** 2)

    if (epoch + 1) % 1000 == 0:
        train_loss = np.mean((y_train - activated_output) ** 2)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

# Função de predição
def predict(phrase, vocab, weights, bias):
    encoded_phrase = encode_sentence(phrase, vocab)
    weighted_sum = np.dot(encoded_phrase, weights) + bias
    output = sigmoid(weighted_sum)
    return output

# Testar o modelo com novas frases
test_phrases = ["oi você", "bom dia para todos", "oi como vai", "até logo", "oi meu caro"]
for phrase in test_phrases:
    prediction = predict(phrase, vocab, weights, bias)
    print(f"Frase: '{phrase}' - Predição: {prediction}")

# Salvar os pesos e bias treinados para uso posterior
np.save('weights.npy', weights)
np.save('bias.npy', bias)



# Carregar os pesos e bias treinados
weights = np.load('weights.npy')
bias = np.load('bias.npy')

# Loop contínuo para entrada do usuário
while True:
    user_input = input("Digite uma frase (ou 'sair' para terminar): ")
    if user_input.lower() == 'sair':
        break
    prediction = predict(user_input, vocab, weights, bias)
    print(f'Predição: {"Contém oi" if prediction >= 0.5 else "Não contém oi"} - Valor: {prediction}')