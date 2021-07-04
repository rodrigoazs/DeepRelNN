import re
from typing import List

from deeprelnn.fol import Literal


class BaseProver:
    def __init__(
            self,
            pos: List[str],
            neg: List[str],
            facts: List[str]
    ) -> None:
        self.pos = self._compile(pos)
        self.neg = self._compile(neg)
        self.facts = self._compile(facts)

    def _compile(self, data: List[str]) -> dict:
        pass

    def _get_literal(self, literal_string: str) -> tuple:
        literals = re.match(
            "([a-zA-Z0-9\_]*)\(([a-zA-Z0-9\,\s\_]*)\)\.", literal_string
        )
        predicate, arguments = literals.groups()
        arguments = re.sub("\s", "", arguments).split(",")
        return predicate, arguments

    def prover(self, head_mapping: dict, clause: List[Literal]) -> List:
        raise NotImplementedError
