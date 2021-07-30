import pandas as pd

from deeprelnn.fol import Constant, Variable
from deeprelnn.prover.base import BaseProver


class Prover(BaseProver):
    def __init__(self, facts):
        super().__init__(facts)

    def _compile(self, data):
        data_dict = {}
        for item in data:
            if isinstance(item, tuple):
                weight, predicate, arguments = item
            else:
                weight, predicate, arguments = self._get_literal(item)
            data_dict.setdefault(predicate, []).append(arguments + [weight])
        for key in data_dict.keys():
            value = data_dict[key]
            data_dict[key] = pd.DataFrame(
                value,
                columns=[
                    "{}_{}".format(key, i)
                    for i in range(len(value[0]) - 1)
                ] + ["weight"],
            )
        return data_dict

    def update_data(self, data):
        self.facts.update(self._compile(data))

    def _satisfy(self, head_mapping, clause):
        last_mapping = head_mapping.copy()
        proved_literals = [0.0] * len(clause)
        for index, literal in enumerate(clause):
            literal_mapping = {}
            for i, argument in enumerate(literal.arguments):
                if type(argument) == Constant:
                    literal_mapping[i] = [argument.name]
                if type(argument) == Variable:
                    if (
                        argument.name != "_"
                        and last_mapping.get(argument.name) is not None  # noqa: E501
                    ):
                        literal_mapping[i] = last_mapping.get(argument.name)
            if literal.predicate.name not in self.facts:
                return proved_literals, last_mapping
            df = self.facts[literal.predicate.name]
            for i, mapping in literal_mapping.items():
                df = df[df[
                    "{}_{}".format(literal.predicate.name, i)].isin(mapping)
                ]
            if not len(df):
                return proved_literals, last_mapping
            for i, argument in enumerate(literal.arguments):
                if type(argument) == Variable and argument.name != "_":
                    last_mapping[argument.name] = set(df[
                        "{}_{}".format(literal.predicate.name, i)
                    ].values.tolist())
            proved_literals[index] = df["weight"].mean()
        return proved_literals, last_mapping

    def prove(self, head_mapping, clause, ignore_weights=False):
        if ignore_weights:
            proved, _ = self._satisfy(head_mapping, clause)
            return proved
        else:
            _, last_mapping = self._satisfy(head_mapping, clause)
            # update weights
            proved_literals, _ = self._satisfy(last_mapping, clause)
        return proved_literals
