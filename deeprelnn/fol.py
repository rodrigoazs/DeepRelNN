import re
from typing import List


class Term:
    def __init__(self, name: str):
        self.name = name

    def is_grounded(self):
        return not self.contains_variables()


class Variable(Term):
    def __init__(self, name: str):
        super().__init__(name)

    def contains_variables(self):
        return True

    def __repr__(self):
        return "Variable({})".format(self.name)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return repr(self) == repr(other)


class Constant(Term):
    def __init__(self, name: str):
        name = re.sub('"', "", name)
        super().__init__(name)

    def contains_variables(self):
        return False

    def __repr__(self):
        return "Constant({})".format(self.name)

    def __str__(self):
        return '"{}"'.format(self.name)

    def __eq__(self, other):
        return repr(self) == repr(other)


class Predicate:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Predicate({})".format(self.name)

    def __str__(self):
        return "{}".format(self.name)

    def __eq__(self, other):
        return repr(self) == repr(other)


class Atom:
    def __init__(
        self,
        predicate: Predicate,
        arguments: List[Term] = [],
        weight: float = 1.0
    ):
        self.predicate = predicate
        self.arguments = arguments
        self.weight = weight


class Literal(Atom):
    def __init__(self, predicate: Predicate, arguments: List[Term] = []):
        super().__init__(predicate, arguments)

    def __repr__(self):
        return "Literal(Predicate({})({}))".format(
            self.predicate.name,
            ", ".join([repr(argument) for argument in self.arguments]),
        )

    def __str__(self):
        return "{}({})".format(
            self.predicate.name,
            ", ".join([str(argument) for argument in self.arguments]),
        )

    def __eq__(self, other):
        return repr(self) == repr(other)
