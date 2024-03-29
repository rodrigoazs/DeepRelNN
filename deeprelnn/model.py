import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from deeprelnn.factory import ClauseFactory
from deeprelnn.parser import get_literal
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
        allow_recursion: bool = True,
        is_regression: bool = False,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: int = 0,
        # predicate_ratio: float = 0.5,
        # sample_ratio: float = 0.5,
    ):
        """Initialize a DeepRDN
        Args:
            background (Background, optional): Background knowledge with
                respect to the database. Defaults to None.
            target (str, optional): Target predicate to learn. Defaults
                to "None".
            number_of_clauses (int, optional): Maximum number of clauses
                in the tree (i.e. maximum number of leaves).
                Defaults to 4.
            number_of_cycles (int, optional): Maximum number of times
                the code will loop to learn clauses, increments even
                if no new clauses are learned. Defaults to 100.
            predicate_ratio (float, optional): Proportion of
                considering a predicate in the search space.
                Defaults to 0.5.
            sample_ratio (float, optional): Proportion of considering
                a sample when training. Defaults to 0.5.
        """
        self.background = background
        self.target = target
        self.number_of_literals = number_of_literals
        self.number_of_clauses = number_of_clauses
        self.number_of_cycles = number_of_cycles
        self.allow_recursion = allow_recursion
        self.is_regression = is_regression
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        # self.predicate_ratio = predicate_ratio
        # self.sample_ratio = sample_ratio
        self.clauses_ = None
        self.estimator_ = None

    def _check_params(self):
        if self.target == "None":
            raise ValueError(
                "target must be set, cannot be {0}".format(self.target)
            )
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

    def fit(self, facts, X):
        """Learn structure and parameters
        """
        # check parameters
        self._check_params()

        # generate clauses
        factory = ClauseFactory(
            self.background,
            facts,
            self.target,
            max_literals=self.number_of_literals,
            max_cycles=self.number_of_cycles,
            allow_recursion=self.allow_recursion)
        self.clauses_ = [
            factory.get_clause()
            for _ in range(self.number_of_clauses)
        ]

        # compile feature and target vectors
        X_train, y_train = self._prove(facts, X)
        X_train = self._get_X(X_train)
        y_train = self._get_y(y_train)

        # get estimator and fit it
        model, params = self._get_estimator(X_train)
        model.fit(X_train, y_train, **params)
        self.estimator_ = model

        return self

    def predict_proba(self, facts, X):
        self._check_is_fitted()
        X_pred, _ = self._prove(facts, X)
        X_pred = self._get_X(X_pred)
        return self.estimator_.predict(X_pred)[:, 1]

    def _prove(self, facts, samples):
        X = []
        y = []
        prover = Prover(facts)
        for sample in samples:
            weight, predicate, arguments = get_literal(sample)
            # check predicate is not target
            if predicate != self.target:
                raise ValueError("Sample predicate is not target")
            head_mapping = {
                chr(65 + index): [argument]
                for index, argument in enumerate(arguments)
            }
            y.append(weight)
            sample_features = []
            for clause in self.clauses_:
                features = prover.prove(head_mapping, clause)
                sample_features.extend(features)
            X.append(sample_features)
        return X, y

    def _get_X(self, X):
        X = np.array(X)
        return X

    def _get_y(self, y):
        if self.is_regression:
            y = np.array(y)
        else:
            y = np.array(
                [
                    [0.0, 1.0] if weight == 1.0 else [1.0, 0.0]
                    for weight in y
                ]
            )
        return y

    def _get_estimator(self, X_train):
        # define the estimator
        model = Sequential()
        model.add(Dropout(0.5, input_shape=(X_train.shape[1],)))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(2, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        # define training params
        params = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
        }

        return model, params

        # Using MLPClassifier
        # from sklearn.neural_network import MLPClassifier
        # # define the estimator
        # model = MLPClassifier(random_state=1, max_iter=300)

        # # define training params
        # params = {
        # }

        return model, params


class NotFittedError(ValueError, AttributeError):
    pass
