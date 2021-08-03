import logging

import numpy as np

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
        allow_recursion,
        criterion,
        strategy,
        random_state
    ):
        self.target = target
        self.modes = modes
        self.constants = constants
        self.max_literals = max_literals
        self.max_predicates = max_predicates
        self.min_examples_learn = min_examples_learn
        self.allow_recursion = allow_recursion
        self.criterion = criterion
        self.strategy = strategy
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
        leaves = []
        best_impurity = np.inf

        logging.info(
            "Starting learner"
        )

        while True:
            # resample predicates in modes
            if self.allow_recursion:
                predicates = list({predicate for predicate, *_ in self.modes})
            else:
                predicates = list({
                    predicate for predicate, *_ in self.modes
                    if predicate != self.target
                })
            predicates.sort()
            self.random_state.shuffle(predicates)
            predicates_to_consider = predicates[:self.max_predicates]
            modes = [
                [predicate, *arguments] for predicate, *arguments
                in self.modes if predicate in predicates_to_consider
            ]

            logging.info(
                "Predicates to consider: {}".format(
                    ", ".join(predicates_to_consider)
                )
            )

            # get potentials
            potentials = LiteralFactory(
                modes,
                self.constants,
                head_variables,
                body_variables
            ).potential_literals()
            # avoid set randomness
            potentials.sort()
            self.random_state.shuffle(potentials)

            logging.info(
                "Number of candidates generated: {}".format(len(potentials))
            )

            if not potentials:
                logging.info(
                    "Stopping due to empty candidates"
                )
                break

            # track potentials
            tuple_potentials = []
            proved_potentials = {}
            impurities = {}

            for potential in potentials:
                proved = []
                true = []
                for i in range(len(examples_)):
                    weight, _, arguments = examples_[i]
                    head_mapping = {
                        chr(65 + index): [argument]
                        for index, argument in enumerate(arguments)
                    }
                    true.append(weight)

                    # if recursion allowed
                    if self.allow_recursion:
                        target_data = examples_[:i] + examples_[i + 1:]
                        prover.update_data(target_data)

                    prove = prover.prove(
                        head_mapping,
                        clause + [potential],
                        ignore_weights=True
                    )  # prove for each example
                    proved.append(prove[-1])  # only last literal is important

                # calculate impurity
                impurity = self.criterion.literal_impurity(
                    proved,
                    true
                )
                impurities[potential] = impurity

                logging.info(
                    "Candidate: {}, Impurity: {}".format(
                        potential,
                        impurity
                    )
                )

                # append results
                tuple_potentials.append((potential, impurity))
                proved_potentials[potential] = proved

            # select best
            best = self.strategy.select_literal(
                tuple_potentials,
                self.random_state
            )

            if not best:
                logging.info("Stopping due to no best candidate found")
                break

            logging.info(
                "Best candidate: {}".format(best)
            )

            # impurity did not improve
            if impurities[best] >= best_impurity:
                logging.info(
                    "Stopping due to no improve in impurity"
                )
                break

            # get new body variables
            body_variables = self._get_new_body_variables(best, body_variables)

            # refilter examples where proved is 1
            proved = proved_potentials[best]
            examples_ = [
                example for example, proved
                in zip(examples_, proved) if proved == 1.0
            ]
            n_examples = len(examples)  # recalculate n_examples

            # calculate leaves
            zipped_proved = list(zip(proved, true))
            pos = np.array([j for p, j in zipped_proved if p == 1.0]).mean()
            neg = np.array([j for p, j in zipped_proved if p == 0.0]).mean()
            leaves.append((pos, neg))

            # add 1 literal
            n_literals += 1
            clause.append(best)

            logging.info(
                "Best literal {} added".format(best)
            )

            # check stop condition
            if n_literals >= self.max_literals or \
               n_examples < self.min_examples_learn or \
               np.sum(true) == len(examples_):
                logging.info(
                    "Stopping due to stopping criteria"
                )
                break

        clause = Clause(clause)
        logging.info(
            "Clause learned: {}".format(clause)
        )

        return clause, leaves
