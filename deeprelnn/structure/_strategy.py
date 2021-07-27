from operator import itemgetter

import numpy as np


class Strategy:
    @staticmethod
    def select_literal(literals, random_state):
        pass


class Best(Strategy):
    @staticmethod
    def select_literal(literals, random_state):
        best = min(literals, key=itemgetter(1))[0]
        return best


class Roulette(Strategy):
    @staticmethod
    def select_literal(literals, random_state):
        lit = [
            literal for literal, impurity
            in literals if not np.isinf(impurity)
        ]
        imp = np.array([
            impurity for _, impurity
            in literals if not np.isinf(impurity)
        ])
        imp = -2 * imp + 1
        imp = imp / imp.sum()
        if not lit:
            return None
        selected = random_state.choice(lit, p=imp)
        return selected
