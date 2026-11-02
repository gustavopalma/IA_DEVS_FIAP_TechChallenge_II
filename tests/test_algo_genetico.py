
import numpy as np
from src.algo_genetico import algoritmo_genetico


def test_algoritmo_genetico():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(60, 8))
    y = np.array([0, 1] * 30)

    melhor, fitness, historico = algoritmo_genetico(
        X,
        y,
        tamanho_populacao=6,
        geracoes=3,
        taxa_mutacao=0.2
    )

    assert melhor is not None
    assert 0 <= fitness <= 1
    assert len(historico) == 3

    assert 2 <= melhor.max_depth <= 15
    assert 2 <= melhor.min_samples_split <= 20
    assert 1 <= melhor.min_samples_leaf <= 10
    assert melhor.criterion in (0, 1)
