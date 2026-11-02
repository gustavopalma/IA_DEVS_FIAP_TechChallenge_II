import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

class Hiperparametros:
    def __init__(self, max_depth=0, min_samples_split=0, min_samples_leaf=0, criterion=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

    def __str__(self):
        return (
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"criterion={self.criterion}"
        )

def iniciar_individuo():
    return Hiperparametros(random.randint(2, 15), random.randint(2, 20), random.randint(2, 10), random.randint(0, 1))

def iniciar_populacao(tamanho):
    pop = []

    for _ in range(tamanho):
        pop.append(iniciar_individuo())

    return pop

def fitness(individuo, X_train, y_train):

    criterio = "gini" if individuo.criterion == 0 else "entropy"

    modelo = DecisionTreeClassifier(max_depth=individuo.max_depth, min_samples_split=individuo.min_samples_split,
        min_samples_leaf=individuo.min_samples_leaf, criterion=criterio, random_state=42)

    validacao = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    resultados = cross_val_score(modelo, X_train, y_train, cv=validacao, scoring="f1", n_jobs=1)

    return resultados.mean()
def selecionar_torneio(populacao, notas, tamanho_torneio=3):

    indices = random.sample(
        range(len(populacao)),
        min(tamanho_torneio, len(populacao))
    )

    vencedor = max(indices, key=lambda i: notas[i])

    return populacao[vencedor]

def crossover(pai1, pai2):

    genes1 = list(vars(pai1).values())
    genes2 = list(vars(pai2).values())

    ponto = random.randint(1, len(genes1) - 1)

    filho1 = Hiperparametros(
        *(genes1[:ponto] + genes2[ponto:])
    )

    filho2 = Hiperparametros(
        *(genes2[:ponto] + genes1[ponto:])
    )

    return filho1, filho2

def mutacao(individuo, taxa_mutacao=0.1):

    if random.random() < taxa_mutacao:

        gene = random.randint(0, 3)

        if gene == 0:
            individuo.max_depth = random.randint(2, 15)

        elif gene == 1:
            individuo.min_samples_split = random.randint(2, 20)

        elif gene == 2:
            individuo.min_samples_leaf = random.randint(1, 10)

        else:
            individuo.criterion = random.randint(0, 1)

    return individuo

def algoritmo_genetico(
    X_train,
    y_train,
    tamanho_populacao=10,
    geracoes=10,
    taxa_mutacao=0.1
):

    if tamanho_populacao < 2 or geracoes < 1:
        raise ValueError("População >= 2 e gerações >= 1 são necessárias")

    populacao = iniciar_populacao(tamanho_populacao)
    historico = []

    melhor_global = None
    melhor_nota = -1

    for geracao in range(geracoes):

        notas = [
            fitness(individuo, X_train, y_train)
            for individuo in populacao
        ]

        indice_melhor = max(
            range(len(populacao)),
            key=lambda i: notas[i]
        )

        melhor_geracao = populacao[indice_melhor]

        if notas[indice_melhor] > melhor_nota:
            melhor_nota = notas[indice_melhor]
            melhor_global = Hiperparametros(
                **vars(melhor_geracao)
            )

        historico.append(melhor_nota)

        print(
            f"Geração {geracao + 1:02d} | "
            f"Melhor fitness: {melhor_nota:.4f}"
        )

        # Elitismo: preserva o melhor indivíduo da geração
        nova_populacao = [
            Hiperparametros(**vars(melhor_geracao))
        ]

        while len(nova_populacao) < tamanho_populacao:

            pai1 = selecionar_torneio(populacao, notas)
            pai2 = selecionar_torneio(populacao, notas)

            filho1, filho2 = crossover(pai1, pai2)

            mutacao(filho1, taxa_mutacao)
            mutacao(filho2, taxa_mutacao)

            nova_populacao.append(filho1)

            if len(nova_populacao) < tamanho_populacao:
                nova_populacao.append(filho2)

        populacao = nova_populacao

    return melhor_global, melhor_nota, historico