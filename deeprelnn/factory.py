from itertools import product

from deeprelnn.fol import Literal, Predicate, Variable


class VariableFactory:
    def __init__(self, initial_variables=[]):
        self.variables = self._set_initial_variables(initial_variables)
        self._last_variable = 65  # ord("A")

    def _set_initial_variables(self, initial_variables):
        variables_set = set()
        if len(initial_variables):
            for variable in initial_variables:
                variables_set.add(variable.name)
        return variables_set

    def _last_variable_to_string(self):
        if self._last_variable > 90:  # ord("Z")
            return "Var{}".format(self._last_variable - 90)
        return chr(self._last_variable)

    def get_new_variable(self):
        while self._last_variable_to_string() in self.variables:
            self._last_variable += 1
        variable = self._last_variable_to_string()
        self.variables.add(variable)
        self._last_variable += 1
        return Variable(variable)

    def copy(self):
        copied = VariableFactory()
        copied.variables = self.variables.copy()
        copied._last_variable = self._last_variable
        return copied


class LiteralFactory:
    def __init__(
        self,
        modes,
        constants,
        head_variables,
        body_variables={}
    ):
        self._modes = modes
        self._constants = constants
        self._head_variables = head_variables
        self._body_variables = body_variables
        self._variable_factory = self._create_variable_factory(
            head_variables,
            body_variables
        )

    def _create_variable_factory(self, head_variables, body_variables):
        initial_variables = set()
        for variables_dict in [head_variables, body_variables]:
            for _, variables in variables_dict.items():
                for variable in variables:
                    initial_variables.add(variable)
        return VariableFactory(initial_variables)

    def potential_literals(self):
        potential = set()
        for predicate, *arguments in self._modes:
            variables = []
            for mode_type, argument_type in arguments:
                if mode_type == "#":
                    vars = set(
                        self._constants.get(argument_type, [])
                    )
                elif mode_type == "+":
                    vars = set(
                        self._head_variables.get(argument_type, [])
                        + self._body_variables.get(argument_type, [])
                    )
                elif mode_type == "-":
                    new_variable = self._variable_factory \
                        .copy().get_new_variable()
                    vars = set(
                        [new_variable]
                        + self._head_variables.get(argument_type, [])
                        + self._body_variables.get(argument_type, [])
                    )
                elif mode_type == "`":
                    new_variable = self._variable_factory \
                        .copy().get_new_variable()
                    vars = set(
                        [new_variable]
                        + self._body_variables.get(argument_type, [])
                    )
                variables.append(vars)
            for args in product(*variables):
                potential.add(Literal(Predicate(predicate), args))
        return list(potential)
