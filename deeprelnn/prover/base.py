from deeprelnn.parser import get_literal


class BaseProver:
    def __init__(self, pos, neg, facts):
        self.pos = self._compile(pos)
        self.neg = self._compile(neg)
        self.facts = self._compile(facts)

    def _compile(self, data):
        pass

    def _get_literal(self, literal_string):
        return get_literal(literal_string)

    def prover(self, head_mapping, clause):
        raise NotImplementedError
