class Builder:
    def __init__(
        self,
        target,
        modes,
        max_literals,
        max_predicates,
        min_examples_learn,
        criterion,
        strategy,
        is_classification,
        random_state
    ):
        self.target = target
        self.modes = modes
        self.max_literals = max_literals
        self.max_predicates = max_predicates
        self.min_examples_learn = min_examples_learn
        self.criterion = criterion
        self.strategy = strategy
        self.is_classification = is_classification
        self.random_state = random_state

    def build(self, examples, prover):
        pass
