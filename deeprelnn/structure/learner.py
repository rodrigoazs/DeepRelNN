
from abc import ABCMeta, abstractmethod

from deeprelnn.parser import get_literal, get_modes
from deeprelnn.prover.prover import Prover
from deeprelnn.structure import _criterion, _strategy
from deeprelnn.utils import check_random_state

CRITERIA_CLF = {"gini": _criterion.Gini}
CRITERIA_REG = {"mse": _criterion.MSE}

STRATEGIES = {"best": _strategy.Best, "roulette": _strategy.Roulette}


class BaseLearner(metaclass=ABCMeta):
    """Base class for rules learner.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 modes,
                 target,
                 criterion,
                 strategy,
                 max_literals,
                 max_predicates,
                 min_examples_learn,
                 random_state):
        self.modes = modes
        self.target = target
        self.criterion = criterion
        self.strategy = strategy
        self.max_literals = max_literals
        self.max_predicates = max_predicates
        self.min_examples_learn = min_examples_learn
        self.random_state = random_state
        self.modes_ = get_modes(modes)

    def _parse_and_validate_examples(self, examples):
        parsed_examples = []
        for example in examples:
            weight, predicate, arguments = get_literal(example)
            # check predicate is not target
            if predicate != self.target:
                raise ValueError(
                    "Example {} predicate is not target".format(example)
                )
            parsed_examples.append((weight, predicate, arguments))
        return parsed_examples

    def _validate_target(self):
        if self.target is None or not len(self.target):
            raise ValueError(
                "Target predicate must be defined"
            )
        target_modes = [
            arguments for predicate, *arguments
            in self.modes_ if predicate == self.target
        ]
        if not len(target_modes):
            raise ValueError(
                "No modes were defined for target predicate"
            )

    def _validate_modes(self):
        predicate_struct = {}
        for predicate, *arguments in self.modes_:
            mode_struct = str([argument[1] for argument in arguments])
            predicate_struct.setdefault(predicate, set()).add(mode_struct)
        for predicate, mode_struct in predicate_struct.items():
            if len(mode_struct) > 1:
                raise ValueError(
                    "Predicate {} modes have "
                    "inconsistent structure".format(predicate)
                )

    def fit(self, examples, background):
        random_state = check_random_state(self.random_state)  # noqa

        # validate target and modes
        self._validate_target()
        self._validate_modes()

        # compile and validate background
        prover = Prover(background)  # noqa

        # validate examples
        examples = self._parse_and_validate_examples(examples)

        # Determine output settings
        n_examples = len(examples)  # noqa
        n_predicates = set(predicate for predicate, *_ in self.modes_)  # noqa

    def predict(self, examples, background):
        pass

    def extract_features(self, examples, background):
        pass


class LearnerClassifier(BaseLearner):
    """A rule learner for classification.

    Args:
        criterion (str): The function to measure the quality of a literal.
            Supported criterion is "gini" for the Gini impurity.
            Defaults to "gini".
        stategy (str): The strategy used to choose the literal.
            Supported strategies are "best" to choose the best literal and
            "roulette" to choose a random literal considering its quality
            as probability. Defaults to "best".
        max_literals (int, optional): The maximum number of literals
            in a rule. If None, then rule is expanded until it
            reaches minimum examples to learn. Defaults to None.
        max_predicates (Union[int, float], optional): The maximum number
            of predicates to consider when looking for the best literal:
            - If int, then consider `max_predicates` predicates at each
                literal search.
            - If float, then consider `max_predicates` is a fraction and
                `int(max_predicates * n_predicates)` predicates
                are considered at each search.
            - If None, then `max_predicates=n_predicates`.
            Defaults to None.
        min_examples_learn (Union[int, float]): The minimum number
            of exmaples required to expand a new literal:
            - If int, then consider `min_examples_learn` as the
                minimum number.
            - If float, then `min_examples_learn` is a fraction
                and `ceil(min_examples_learn * n_examples)`
                are the minimum number of examples.
            Defaults to 1.
        random_state (int, optional): Controls the randomness of
            the learner. The predicates are always shuffled at
            each literal search, even if ``strategy`` is set to
            ``"best"``. Defaults to None.
    """
    def __init__(self,
                 modes,
                 target,
                 criterion="gini",
                 strategy="best",
                 max_literals=None,
                 max_predicates=None,
                 min_examples_learn=1,
                 random_state=None):
        super().__init__(
            modes=modes,
            target=target,
            criterion=criterion,
            strategy=strategy,
            max_literals=max_literals,
            max_predicates=max_predicates,
            min_examples_learn=min_examples_learn,
            random_state=random_state)
        self.is_classifier = True

    def fit(self, examples, background):
        super().fit(examples, background)
        return self

    def predict_proba(self, examples, background):
        pass


class LearnerRegressor(BaseLearner):
    """A rule learner for classification.

    Args:
        criterion (str): The function to measure the quality of a literal.
            Supported criterion is "mse" for the mean squared error.
            Defaults to "mse".
        stategy (str): The strategy used to choose the literal.
            Supported strategies are "best" to choose the best literal and
            "roulette" to choose a random literal considering its quality
            as probability. Defaults to "best".
        max_literals (int, optional): The maximum number of literals
            in a rule. If None, then rule is expanded until it
            reaches minimum examples to learn. Defaults to None.
        max_predicates (Union[int, float], optional): The maximum number
            of predicates to consider when looking for the best literal:
            - If int, then consider `max_predicates` predicates at each
                literal search.
            - If float, then consider `max_predicates` is a fraction and
                `int(max_predicates * n_predicates)` predicates
                are considered at each search.
            - If None, then `max_predicates=n_predicates`.
            Defaults to None.
        min_examples_learn (Union[int, float]): The minimum number
            of exmaples required to expand a new literal:
            - If int, then consider `min_examples_learn` as the
                minimum number.
            - If float, then `min_examples_learn` is a fraction
                and `ceil(min_examples_learn * n_examples)`
                are the minimum number of examples.
            Defaults to 1.
        random_state (int, optional): Controls the randomness of
            the learner. The predicates are always shuffled at
            each literal search, even if ``strategy`` is set to
            ``"best"``. Defaults to None.
    """
    def __init__(self,
                 modes,
                 target,
                 criterion="mse",
                 strategy="best",
                 max_literals=None,
                 max_predicates=None,
                 min_examples_learn=1,
                 random_state=None):
        super().__init__(
            modes=modes,
            target=target,
            criterion=criterion,
            strategy=strategy,
            max_literals=max_literals,
            max_predicates=max_predicates,
            min_examples_learn=min_examples_learn,
            random_state=random_state)
        self.is_classifier = False

    def fit(self, examples, background):
        super().fit(examples, background)
        return self
