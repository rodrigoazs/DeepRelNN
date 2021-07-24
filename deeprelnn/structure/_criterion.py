import numpy as np


class Criterion:
    @staticmethod
    def literal_impurity(att, true):
        pass


class Gini(Criterion):
    @staticmethod
    def literal_impurity(att, true):
        m = list(zip(att, true))
        pos = [j for p, j in m if p == 1.0]
        neg = [j for p, j in m if p == 0.0]
        gini_pos = 1.0 - (np.sum(pos) / len(pos))**2 \
            - ((len(pos) - np.sum(pos)) / len(pos))**2
        gini_neg = 1.0 - (np.sum(neg) / len(neg))**2 \
            - ((len(neg) - np.sum(neg)) / len(neg))**2
        gini = len(pos) / len(m) * gini_pos + len(neg) / len(m) * gini_neg
        return gini


class MSE(Criterion):
    @staticmethod
    def literal_impurity(att, true):
        m = list(zip(att, true))
        pos = np.array([j for p, j in m if p == 1.0])
        neg = np.array([j for p, j in m if p == 0.0])
        mse_pos = ((pos - pos.mean())**2).mean()
        mse_neg = ((neg - neg.mean())**2).mean()
        mse = len(pos) / len(m) * mse_pos + len(neg) / len(m) * mse_neg
        return mse
