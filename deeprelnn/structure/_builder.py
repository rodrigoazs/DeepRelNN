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
        n_literals = 0
        while True:
            potentials = ["A", "B"]  # LiteralFactory get potential literals
            if not potentials:  # no potential
                break
            tuple_potentials = []
            proved_potentials = {}
            for potential in potentials:
                proved = [1 for i in range(10)]  # Prove for each example
                impurity = self.criterion.literal_impurity(
                    proved,
                    proved
                )  # proved, true classes
                tuple_potentials.append((potential, impurity))
                proved_potentials[potential] = proved
            # select best
            best = self.strategy.select_literal(
                tuple_potentials,
                self.random_state
            )
            proved = proved_potentials[best]
            # refilter examples by proved where is 1
            n_examples = 9999  # recalculate n_examples
            n_literals += 1
            if n_literals >= self.max_literals or \
               n_examples < self.min_examples_learn:
                break
