from deeprelnn.factory import LiteralFactory
from deeprelnn.fol import Clause, Variable


class Builder:
    def __init__(
        self,
        target,
        modes,
        constants,
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
        self.constants = constants
        self.max_literals = max_literals
        self.max_predicates = max_predicates
        self.min_examples_learn = min_examples_learn
        self.criterion = criterion
        self.strategy = strategy
        self.is_classification = is_classification
        self.random_state = random_state

    def _create_head_variables(self):
        # create head mapping
        for predicate, *arguments in self.modes:
            if predicate == self.target:
                head_variables = {}
                for index, (mode, type) in enumerate(arguments):
                    head_variables.setdefault(type, []) \
                        .append(Variable(chr(65 + index)))
                return head_variables
        raise ValueError("No target mode found")

    def _get_new_body_variables(self, best, body_variables):
        # get literal argument types
        for predicate, *arguments in self.modes:
            if best.predicate.name == predicate:
                argument_types = [arg_type for _, arg_type in arguments]
                for variable, arg_type in zip(best.arguments, argument_types):
                    body_variables.setdefault(arg_type, []).append(variable)
                    body_variables[arg_type] = list(
                        set(body_variables[arg_type])
                    )
                return body_variables
        raise ValueError("No best literal mode found")

    def build(self, examples, prover):
        # copy examples
        examples_ = examples.copy()

        # set initial
        head_variables = self._create_head_variables()
        body_variables = {}
        n_literals = 0
        clause = []
        best_impurity = self.criterion.literal_impurity(
            [1.0] * len(examples_),
            [weight for weight, _, _ in examples_]
        )

        print('True body impurity', best_impurity)

        while True:
            potentials = LiteralFactory(
                self.modes,
                self.constants,
                head_variables,
                body_variables
            ).potential_literals()
            if not potentials:
                break

            # track potentials
            tuple_potentials = []
            proved_potentials = {}
            impurities = {}

            for potential in potentials:
                proved = []
                true = []
                for example in examples_:
                    weight, predicate, arguments = example
                    head_mapping = {
                        chr(65 + index): [argument]
                        for index, argument in enumerate(arguments)
                    }
                    true.append(weight)
                    prove = prover.prove(
                        head_mapping,
                        clause + [potential],
                        ignore_weights=True
                    )  # prove for each example
                    proved.append(prove[-1])  # only last literal is important
                print('protential', potential)
                print('proved', proved)
                print('true', true)
                # calculate impurity
                impurity = self.criterion.literal_impurity(
                    proved,
                    true
                )
                impurities[potential] = impurity
                print('impurity', impurity)
                # append results
                tuple_potentials.append((potential, impurity))
                proved_potentials[potential] = proved

            # select best
            best = self.strategy.select_literal(
                tuple_potentials,
                self.random_state
            )
            print('best', best)

            # impurity did not improve
            if impurities[best] >= best_impurity:
                break

            # get new body variables
            body_variables = self._get_new_body_variables(best, body_variables)

            # refilter examples where proved is 1
            proved = proved_potentials[best]
            examples_ = [
                example for example, proved
                in zip(examples, proved) if proved == 1.0
            ]
            n_examples = len(examples)  # recalculate n_examples

            # add 1 literal
            n_literals += 1
            clause.append(best)

            # check stop condition
            if n_literals >= self.max_literals or \
               n_examples < self.min_examples_learn:
                break
        return Clause(clause)
