import re


class Term:
    def __init__(self, name):
        self.name = name

    def is_grounded(self):
        return not self.contains_variables()

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class Variable(Term):
    def __init__(self, name):
        super().__init__(name)

    def contains_variables(self):
        return True

    def __repr__(self):
        return "Variable({})".format(self.name)

    def __str__(self):
        return self.name


class Constant(Term):
    def __init__(self, name):
        name = re.sub('"', "", name)
        super().__init__(name)

    def contains_variables(self):
        return False

    def __repr__(self):
        return "Constant({})".format(self.name)

    def __str__(self):
        return '"{}"'.format(self.name)


class Predicate:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Predicate({})".format(self.name)

    def __str__(self):
        return "{}".format(self.name)

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class Atom:
    def __init__(self, predicate, arguments=[], weight=1.0):
        self.predicate = predicate
        self.arguments = arguments
        self.weight = weight

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

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        return hash(repr(self))


class Literal(Atom):
    def __init__(self, predicate, arguments=[], weight=1.0):
        super().__init__(predicate, arguments, weight=1.0)


class Clause:
    def __init__(self, literals):
        self.literals = literals

    def __str__(self):
        return ", ".join(map(str, self.literals))

    def __repr__(self):
        return "Clause({})".format(
            ", ".join([repr(literal) for literal in self.literals]),
        )

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))
