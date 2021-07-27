import numpy as np
import pytest

from deeprelnn.structure._strategy import Best, Roulette


def test_best_select_literal():
    literals = [("A", 0.2), ("B", 0.6)]
    best = Best.select_literal(literals, None)
    assert best == "A"


def test_roulette_select_literal():
    literals = [("A", 0.5), ("B", 0.1)]
    random_state = np.random.RandomState(1)
    selected = Roulette.select_literal(literals, random_state)
    assert selected == "B"


@pytest.mark.parametrize("impurities", [
    (0.5, 0),
    (0.4, 24)
])
def test_roulette_select_literal(impurities):
    literals = [("A", impurities[0]), ("B", 0.1)]
    random_state = np.random.RandomState(1)
    count_A = 0
    for i in range(100):
        selected = Roulette.select_literal(literals, random_state)
        if selected == "A":
            count_A +=1
    assert count_A == impurities[1]
