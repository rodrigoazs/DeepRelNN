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
        # TODO: add this to literals generated
        self._n_predicate_examples = None

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

    def __lt__(self, other):
        left_consts = sum([
            1 for argument in self.arguments
            if isinstance(argument, Constant)
        ])
        right_consts = sum([
            1 for argument in other.arguments
            if isinstance(argument, Constant)
        ])
        left_len = len(self.arguments)
        right_len = len(other.arguments)
        left_result = left_len - left_consts
        right_result = right_len - right_consts
        if left_result == right_result:
            if left_len == right_len:
                left_n_predicate_examples = self._n_predicate_examples \
                    if self._n_predicate_examples \
                    else float("inf")
                right_n_predicate_examples = other._n_predicate_examples \
                    if other._n_predicate_examples \
                    else float("inf")
                return left_n_predicate_examples < right_n_predicate_examples
            else:
                return left_consts > right_consts
        else:
            return left_result < right_result

    def __eq__(self, other):
        return repr(self) == repr(other)

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
