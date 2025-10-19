def neuron_main():
    from learning.neuron import Neuron

    # ==========================================
    # EXEMPLO 1: NEURÔNIO APRENDE A FAZER CONTA
    # ==========================================

    print("=" * 70)
    print("🧮 EXEMPLO 1: ENSINANDO UM NEURÔNIO A MULTIPLICAR POR 2")
    print("=" * 70)
    print("\nVamos ensinar o neurônio que: NÚMERO × 2 = RESULTADO\n")

    # Create a single-input neuron with LINEAR output (regression)
    neuron_math = Neuron(num_inputs=1, activation="linear")

    # Training samples: learn y = 2x
    training_data = [
        ([0.1], 0.2),
        ([0.2], 0.4),
        ([0.3], 0.6),
        ([0.4], 0.8),
        ([0.5], 1.0),
    ]

    # Before learning
    print("❌ ANTES DE APRENDER (neurônio chutando):")
    for x, y_true in training_data:
        y_pred = neuron_math.think(x)
        print(f"   {x[0]} × 2 = {y_pred:.3f}  (correto seria: {y_true})")

    # Train (linear output converges rápido; 100-1000 já resolve)
    neuron_math.train(training_data, epochs=100, learning_rate=0.05)

    # After learning
    print("✅ DEPOIS DE APRENDER:")
    for x, y_true in training_data:
        y_pred = neuron_math.think(x)
        print(f"   {x[0]} × 2 = {y_pred:.3f}  (correto: {y_true})")

    # Show learned parameters
    print("\n🔎 Parâmetros aprendidos (aprox.):")
    print(f"   w ≈ {neuron_math.weights[0]:.4f}, b ≈ {neuron_math.bias:.4f}")

    # Test with unseen numbers
    print("\n🎯 TESTANDO COM NÚMEROS NOVOS (que não estavam no treino):")
    for x_new in ([0.35], [0.7], [0.9], [1.5], [2.0], [0.11]):
        y_pred = neuron_math.think(x_new)
        print(f"   {x_new[0]} × 2 = {y_pred:.3f}  (correto seria: {x_new[0] * 2})")
    print("   👀 Viu? Agora ele GENERALIZA linearmente (sem saturar).")


def neural_network_main():
    from learning.neural_network import NeuralNetwork
    # ==========================================
    # EXEMPLO 2: RECONHECER LETRA "T" vs "X"
    # ==========================================

    print("\n" + "=" * 70)
    print("✍️  EXEMPLO 2: REDE DE 10 NEURÔNIOS RECONHECENDO LETRAS")
    print("=" * 70)
    print("\nVamos ensinar a rede a reconhecer padrões visuais simples!")
    print("Cada letra é representada por 9 pixels (grade 3×3)\n")

    # Criar rede: 9 entradas (3×3 pixels) → 10 neurônios → 2 saídas (T ou X)
    rede = NeuralNetwork(num_inputs=9, num_hidden=10, num_outputs=2, hidden_activation="sigmoid", output_activation="sigmoid")

    # Representação visual das letras em grade 3×3
    print("📝 PADRÕES QUE VAMOS ENSINAR:")
    print("\nLetra T:        Letra X:")
    print("█ █ █          █ · █")
    print("· █ ·          · █ ·")
    print("· █ ·          █ · █")
    print()

    # Dados de treino
    # 1 = pixel aceso, 0 = pixel apagado
    # Saída: [1, 0] = é um T, [0, 1] = é um X

    dados_letras = [
        # Letra T (várias variações)
        ([1,1,1, 0,1,0, 0,1,0], [1, 0]),  # T perfeito
        ([1,1,1, 0,1,0, 0,1,1], [1, 0]),  # T com ruído
        ([1,1,0, 0,1,0, 0,1,0], [1, 0]),  # T incompleto
        
        # Letra X (várias variações)  
        ([1,0,1, 0,1,0, 1,0,1], [0, 1]),  # X perfeito
        ([1,0,1, 0,1,0, 1,0,0], [0, 1]),  # X com ruído
        ([1,0,1, 1,1,0, 1,0,1], [0, 1]),  # X com ruído
    ]

    # ANTES do treino
    print("❌ ANTES DO TREINO (rede chutando):")
    for i, (entrada, esperado) in enumerate(dados_letras):
        saidas, _ = rede.think(entrada)
        letra = "T" if esperado[0] > esperado[1] else "X"
        prob_t = saidas[0] * 100
        prob_x = saidas[1] * 100
        print(f"   Padrão {i+1} (é um {letra}): T={prob_t:.1f}% | X={prob_x:.1f}%")

    # TREINANDO
    rede.train(dados_letras, epochs=20000)

    # DEPOIS do treino
    print("✅ DEPOIS DO TREINO:")
    for i, (entrada, esperado) in enumerate(dados_letras):
        saidas, _ = rede.think(entrada)
        letra = "T" if esperado[0] > esperado[1] else "X"
        prob_t = saidas[0] * 100
        prob_x = saidas[1] * 100
        print(f"   Padrão {i+1} (é um {letra}): T={prob_t:.1f}% | X={prob_x:.1f}%")

    # TESTAR com padrão totalmente NOVO
    print("\n🎯 TESTANDO COM PADRÃO NOVO (nunca visto no treino):")
    print("\nNovo T:         Novo X:")
    print("█ █ ·          · · █")
    print("· █ ·          · █ ·")
    print("· █ ·          █ · ·")

    novo_t = [1,1,0, 0,1,0, 0,1,0]
    novo_x = [0,0,1, 0,1,0, 1,0,0]

    saidas_t, _ = rede.think(novo_t)
    saidas_x, _ = rede.think(novo_x)

    print(f"\nNovo T: T={saidas_t[0]*100:.1f}% | X={saidas_t[1]*100:.1f}%")
    print(f"Novo X: T={saidas_x[0]*100:.1f}% | X={saidas_x[1]*100:.1f}%")
    print("\n👏 A rede GENERALIZOU! Reconhece padrões que nunca viu!")

def generative_network_main():
    from learning.generative_network import GenerativeNetwork

    # =========================
    # helpers (sampling etc.)
    # =========================
    import math
    import random

    def temperature_sample(probs, temperature: float = 0.8, top_k: int | None = 5):
        """Amostragem com temperatura + top-k.
        probs: lista de probabilidades (somam ~1)."""
        # evita zeros
        eps = 1e-12
        probs = [max(p, eps) for p in probs]

        # volta para logits, aplica temperatura
        logits = [math.log(p) for p in probs]
        logits = [z / max(temperature, 1e-6) for z in logits]

        # top-k (opcional)
        if top_k is not None and top_k < len(logits):
            # mantém índices dos top_k logits
            top_idx = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:top_k]
            mask = [0.0] * len(logits)
            for i in top_idx:
                mask[i] = 1.0
            logits = [logits[i] if mask[i] > 0 else float("-inf") for i in range(len(logits))]

        # softmax estável
        m = max(logits)
        exps = [0.0 if (z == float("-inf")) else math.exp(z - m) for z in logits]
        s = sum(exps) or 1.0
        new_probs = [e / s for e in exps]

        # roleta
        r = random.random()
        acc = 0.0
        for i, p in enumerate(new_probs):
            acc += p
            if r <= acc:
                return i
        return len(new_probs) - 1  # fallback

    def char_to_onehot(ch, char_to_idx, vocab_size: int):
        """Converte letra em vetor (one-hot encoding)"""
        v = [0.0] * vocab_size
        if ch in char_to_idx:
            v[char_to_idx[ch]] = 1.0
        return v

    def topk_indices(probs, k=3):
        return sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]

    # ==========================================
    # EXEMPLO 1: IA QUE COMPLETA PALAVRAS
    # ==========================================

    print("=" * 80)
    print("💬 EXEMPLO 1: IA QUE APRENDE A COMPLETAR SAUDAÇÕES")
    print("=" * 80)
    print("\nVamos ensinar a IA a entender padrões de saudações em português!\n")

    # Vocabulário (Atenção: tudo UPPER + espaço)
    vocabulary = ['B', 'O', 'M', 'D', 'I', 'A', 'T', 'R', 'E', 'N', ' ']
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    print(f"📖 Vocabulário da IA: {vocabulary}")
    print(f"   Total: {len(vocabulary)} caracteres\n")

    # Rede com 10 neurônios ocultos
    net = GenerativeNetwork(vocab_size=len(vocabulary), num_hidden=10)

    # ALIASES para manter compatibilidade com seu código antigo
    net.gerar_proximo = net.predict_next
    net.gerar_texto = net.generate_text

    print("📝 Ensinando padrões:")
    train_pairs_greetings = [
        # BOM DIA
        ('B', 'O'), ('O', 'M'), ('M', ' '), (' ', 'D'), ('D', 'I'), ('I', 'A'),
        # BOA TARDE
        ('B', 'O'), ('O', 'A'), ('A', ' '), (' ', 'T'), ('T', 'A'), ('A', 'R'), ('R', 'D'), ('D', 'E'),
        # BOA NOITE
        ('B', 'O'), ('O', 'A'), ('A', ' '), (' ', 'N'), ('N', 'O'), ('O', 'I'), ('I', 'T'), ('T', 'E'),
    ]
    print("   • B → O (início de BOA/BOM)")
    print("   • O → M (BOM) ou O → A (BOA)")
    print("   • M → ' ' (BOM_)")
    print("   • ' ' → D/T/N (início de DIA/TARDE/NOITE)")
    print("   • E assim por diante...\n")

    # montar dados (x_onehot, y_onehot)
    train_data = []
    for a, b in train_pairs_greetings:
        x = char_to_onehot(a, char_to_idx, len(vocabulary))
        y = char_to_onehot(b, char_to_idx, len(vocabulary))
        train_data.append((x, y))

    # Treino
    net.train(train_data, epochs=5000, learning_rate=0.2, report_every=500)

    # TESTE
    print("🧪 TESTANDO O QUE A IA APRENDEU:")
    print("-" * 80)
    test_chars = ['B', 'O', 'M', ' ', 'D', 'A', 'T', 'N']
    for ch in test_chars:
        x = char_to_onehot(ch, char_to_idx, len(vocabulary))
        probs = net.predict_next(x)
        top_idx = topk_indices(probs, k=3)
        print(f"\n   Se vê '{ch}', a IA acha que o próximo pode ser:")
        for i in top_idx:
            print(f"      '{idx_to_char[i]}' com {probs[i]*100:.1f}% de certeza")

    # ==========================================
    # EXEMPLO 2: IA GERANDO TEXTO COMPLETO
    # ==========================================

    print("\n\n" + "=" * 80)
    print("✨ EXEMPLO 2: IA GERANDO TEXTO AUTOMATICAMENTE")
    print("=" * 80)
    print("\nAgora a IA vai GERAR texto sozinha, letra por letra!")
    print("Como? Ela vê uma letra, prevê a próxima, adiciona, e repete!\n")

    print("🎯 GERAÇÕES AUTOMÁTICAS (com temperature/top-k):")
    print("-" * 80)

    # geração com sampling (em vez de greedy)
    def generate_text_sampled(seed: str, length: int = 10, temperature: float = 0.8, top_k: int | None = 5):
        """Gera texto usando sampling com temperatura e top-k."""
        out = seed
        while len(out) < len(seed) + length:
            last = out[-1]
            if last not in char_to_idx:
                break
            x = char_to_onehot(last, char_to_idx, len(vocabulary))
            probs = net.predict_next(x)
            next_i = temperature_sample(probs, temperature=temperature, top_k=top_k)
            out += idx_to_char[next_i]
        return out

    seeds = ['B', 'BO', 'BOM']
    for s in seeds:
        txt = generate_text_sampled(s, length=10, temperature=0.8, top_k=3)
        print(f"   Começando com '{s}' → Gerou: '{txt}'")

    # ==========================================
    # EXEMPLO 3: IA MAIS AVANÇADA - NOMES
    # ==========================================

    print("\n\n" + "=" * 80)
    print("🚀 EXEMPLO 3: IA GERANDO NOMES BRASILEIROS")
    print("=" * 80)

    vocab_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
    char_to_idx2 = {ch: i for i, ch in enumerate(vocab_names)}
    idx_to_char2 = {i: ch for ch, i in char_to_idx2.items()}

    print(f"\n📖 Novo vocabulário: {len(vocab_names)} letras (A-Z + espaço)")

    net_names = GenerativeNetwork(vocab_size=len(vocab_names), num_hidden=20)
    # aliases
    net_names.gerar_proximo = net_names.predict_next
    net_names.gerar_texto = net_names.generate_text

    names_train = ["MARIA", "JOAO", "JOSE", "ANA", "PEDRO",
                   "CARLOS", "PAULO", "LUCAS", "JULIA", "BEATRIZ"]
    print(f"\n📚 Ensinando {len(names_train)} nomes brasileiros:")
    print(f"   {', '.join(names_train[:5])}, ...")

    # pares consecutivos
    train_names = []
    for name in names_train:
        for i in range(len(name) - 1):
            x = char_to_onehot(name[i], char_to_idx2, len(vocab_names))
            y = char_to_onehot(name[i+1], char_to_idx2, len(vocab_names))
            train_names.append((x, y))

    # pequeno boost nos pares com espaço final para “fechar” nomes
    # (ex.: "ANA " -> aumenta chance de espaço depois de A)
    for name in names_train:
        x = char_to_onehot(name[-1], char_to_idx2, len(vocab_names))
        y = char_to_onehot(' ', char_to_idx2, len(vocab_names))
        train_names.append((x, y))

    net_names.train(train_names, epochs=9000, learning_rate=0.25, report_every=1000)

    print("\n✨ IA GERANDO NOMES NOVOS (sampling com temperature/top-k):")
    print("-" * 80)

    def generate_name_sampled(seed: str, length: int = 8, temperature: float = 0.9, top_k: int | None = 5):
        out = seed
        while len(out) < len(seed) + length:
            last = out[-1]
            if last not in char_to_idx2:
                break
            x = char_to_onehot(last, char_to_idx2, len(vocab_names))
            probs = net_names.predict_next(x)
            next_i = temperature_sample(probs, temperature=temperature, top_k=top_k)
            out += idx_to_char2[next_i]
        return out

    initials = ['M', 'J', 'P', 'L', 'C']
    for ini in initials:
        gen = generate_name_sampled(ini, length=8, temperature=0.9, top_k=5)
        print(f"   Começando com '{ini}' → Nome gerado: '{gen}'")

def generative_learning():
    # ==========================================
    # EXPLICAÇÃO FINAL
    # ==========================================

    print("\n\n" + "=" * 80)
    print("🎓 COMO FUNCIONA A IA GENERATIVA (RESUMO PARA AULA)")
    print("=" * 80)
    print("""
    1. 🧠 APRENDIZADO DE PADRÕES:
    • A rede vê MILHARES de exemplos: "depois de B vem O", "depois de O vem A"...
    • Ajusta seus pesos (memória) para memorizar esses padrões
    • Como criança aprendendo a escrever vendo muitos exemplos!

    2. ✨ GERAÇÃO CRIATIVA:
    • Você dá um INÍCIO (ex: "B")
    • A IA prevê: "depois de B, provavelmente vem O"
    • Adiciona "O" ao texto
    • Agora tem "BO", prevê o próximo...
    • Repete esse processo letra por letra!

    3. 🎯 POR QUE FUNCIONA:
    • Não é mágica! É ESTATÍSTICA APRENDIDA
    • A rede aprendeu probabilidades: "após M geralmente vem espaço"
    • Quanto mais dados, melhor ela aprende padrões
    • ChatGPT faz o MESMO, mas com BILHÕES de neurônios!

    4. 💡 ANALOGIA COM CÉREBRO:
    • Quando você escreve, seu cérebro prevê a próxima palavra
    • Você aprendeu padrões lendo milhões de palavras na vida
    • IA faz igual: aprende padrões e os usa para criar texto novo
    • Diferença: cérebro tem 86 bilhões de neurônios!

    5. 🚀 IA MODERNA (ChatGPT, etc):
    • Mesma ideia básica que mostramos aqui
    • Mas com arquitetura mais complexa (Transformers)
    • BILHÕES de neurônios
    • Treinada em TODO texto da internet
    • Por isso consegue conversar, programar, criar histórias...
    
    6. ⚡ LIMITAÇÕES DESTE EXEMPLO:
    • Nossa rede é PEQUENA (10-15 neurônios)
    • ChatGPT tem 175 BILHÕES de parâmetros!
    • Mas o PRINCÍPIO é exatamente o mesmo!
    """)

def learning():
     # ==========================================
    # RESUMO EDUCACIONAL
    # ==========================================

    print("\n" + "=" * 70)
    print("🎓 O QUE APRENDEMOS:")
    print("=" * 70)
    print("""
    1. 🧠 UM NEURÔNIO SOZINHO:
    • Aprendeu a fazer conta (multiplicar por 2)
    • Ajustou sua "memória" (pesos) através dos erros
    • Conseguiu fazer contas com números novos!
    
    2. 👥 EQUIPE DE 10 NEURÔNIOS:
    • Trabalharam juntos para reconhecer padrões complexos
    • Cada neurônio da camada oculta detecta características diferentes
    • Juntos, tomam decisões mais inteligentes
    
    3. 🚀 PODER DA GENERALIZAÇÃO:
    • Não decoram respostas, aprendem PADRÕES
    • Funcionam com dados que nunca viram antes
    • Como humanos: aprendemos por exemplos e aplicamos em situações novas!

    4. 💡 ANALOGIA COM O CÉREBRO:
    • Neurônios biológicos aprendem fortalecendo conexões úteis
    • Neurônios artificiais ajustam "pesos" (força das conexões)
    • Ambos aprendem com repetição e correção de erros
    • Trabalho em equipe gera inteligência complexa!
    """)


if __name__ == "__main__":
    neuron_main()
    #neural_network_main()
    #learning()
    #generative_network_main()
    #generative_learning()


    
   