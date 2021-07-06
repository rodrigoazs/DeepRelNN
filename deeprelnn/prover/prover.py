import pandas as pd

from deeprelnn.fol import Constant, Variable
from deeprelnn.prover.base import BaseProver


class Prover(BaseProver):
    def __init__(self, pos, neg, facts):
        super().__init__(pos, neg, facts)

    def _compile(self, data):
        data_dict = {}
        for item in data:
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

    def prove(self, head_mapping, clause):
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
                return proved_literals
            df = self.facts[literal.predicate.name]
            for i, mapping in literal_mapping.items():
                df = df[df[
                    "{}_{}".format(literal.predicate.name, i)].isin(mapping)
                ]
            if not len(df):
                return proved_literals
            for i, argument in enumerate(literal.arguments):
                if type(argument) == Variable and argument.name != "_":
                    last_mapping[argument.name] = df[
                        "{}_{}".format(literal.predicate.name, i)
                    ].values.tolist()
            proved_literals[index] = df["weight"].mean()
        return proved_literals
