# flake8: noqa
import random
import re

import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from deeprelnn.factory import ClauseFactory
from deeprelnn.prover.prover import Prover


class DeepRelNN:
    """Deep Relational Neural Network Estimator
    """
    def __init__(
        self,
        background: None,
        target: str = "None",
        number_of_literals: int = 4,
        number_of_clauses: int = 100,
        number_of_cycles: int = 10,
        predicate_ratio: float = 0.5,
        sample_ratio: float = 0.5,
    ):
        """Initialize a DeepRDN
        Args:
            background (Background, optional): Background knowledge with
                respect to the database. Defaults to None.
            target (str, optional): Target predicate to learn. Defaults to "None".
            number_of_clauses (int, optional): Maximum number of clauses in the
                tree (i.e. maximum number of leaves). Defaults to 4.
            number_of_cycles (int, optional): Maximum number of times the code
                will loop to learn clauses, increments even if no new clauses are
                learned. Defaults to 100.
            predicate_ratio (float, optional): Proportion of considering a predicate
                in the search space. Defaults to 0.5.
            sample_ratio (float, optional): Proportion of considering a sample when
                training. Defaults to 0.5.
        """
        self.background = background
        self.target = target
        self.predicate_ratio = predicate_ratio
        self.sample_ratio = sample_ratio
        self.trees_ = None
        self.clauses_ = None
        self.estimator_ = None

    def _check_params(self):
        if self.target == "None":
            raise ValueError("target must be set, cannot be {0}".format(self.target))
        if not isinstance(self.target, str):
            raise ValueError(
                "target must be a string, cannot be {0}".format(self.target)
            )
        if self.background is None:
            raise ValueError(
                "background must be set, cannot be {0}".format(self.background)
            )

    def _check_is_fitted(self):
        if self.estimator_ is None:
            raise NotFittedError

    def fit(self, facts, X, y):
        """Learn structure and parameters
        """
        # check parameters
        self._check_params()

        # generate clauses
        factory = ClauseFactory(self.background, facts, self.target)
        self.clauses_ = factory.get_clauses()
        
        # compile feature and target vectors
        X_train = self._get_X(X)
        y_train = self._get_y(y)
        
        # get estimator and fit it
        model, params = self._get_estimator(X_train)
        model.fit(X_train, y_train, **params)
        self.estimator_ = model

        return self

    def predict_proba(self, database):
        self._check_is_fitted()
        X_pred = self._get_X(database)
        return self.estimator_.predict(X_pred)[:, 1]

    def _prove(self, facts, X):
        
        X = []
        prover = Prover(facts)
        for sample in X:
            for clause in self.clauses_:
                pass

    def _get_X(self, X):  # noqa: N803, N802
        """X = np.array(
        )
        return X"""
        pass

    def _get_y(self, y):
        y = np.array(
            [
                [0.0, 1.0] if _ is True or _ == 1.0 else [1.0, 0.0]
                for _ in range(len(y))
            ]
        )
        return y

    def _get_estimator(self, X_train):  # noqa: N803
        # define the estimator
        model = Sequential()
        model.add(Dropout(0.5, input_shape=(X_train.shape[1],)))
        model.add(Dense(10))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Dense(2, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        # define training params
        params = {
            'epochs': 25,
            'batch_size': 32,
            'verbose': 0,
        }

        return model, params


class NotFittedError(ValueError, AttributeError):
    pass
