import numpy as np

from deeprelnn.structure._strategy import Best, Roulette


def test_best_select_literal():
    literals = [("A", 0.2), ("B", 0.6)]
    best = Best.select_literal(literals, None)
    assert best == "A"


def test_roulette_select_literal():
    literals = [("A", 0.9), ("B", 0.1)]
    random_state = np.random.RandomState(1)
    selected = Roulette.select_literal(literals, random_state)
    assert selected == "B"
