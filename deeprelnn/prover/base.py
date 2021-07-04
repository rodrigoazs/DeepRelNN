import re


class BaseProver:
    def __init__(self, pos, neg, facts):
        self.pos = self._compile(pos)
        self.neg = self._compile(neg)
        self.facts = self._compile(facts)

    def _compile(self, data):
        pass

    def _get_literal(self, literal_string):
        literals = re.match(
            "([a-zA-Z0-9\_]*)\(([a-zA-Z0-9\,\s\_]*)\)\.", literal_string
        )
        predicate, arguments = literals.groups()
        arguments = re.sub("\s", "", arguments).split(",")
        return predicate, arguments

    def prover(self, head_mapping, clause):
        raise NotImplementedError
