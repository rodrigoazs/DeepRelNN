import random

from deeprelnn.fol import Constant, Literal, Predicate, Variable
from deeprelnn.parser import get_constants, get_modes


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


class ClauseFactory:
    def __init__(
        self,
        modes,
        facts,
        target,
        max_literals=4,
        max_cycles=10,
        allow_recursion=True
    ):
        self._modes = get_modes(modes)
        self._constants = get_constants(self._modes, facts)
        self._target = target
        self._reset_variables()
        self._max_literals = max_literals
        self._max_cycles = max_cycles
        self._allow_recursion = allow_recursion

    def _get_potential_modes_indexes(self, head_variables, body_variables):
        potential_modes = []
        for index, mode in enumerate(self._modes):
            if not self._allow_recursion and mode[0] == self._target:
                continue
            potential = True
            for mode, type in mode[1:]:
                if mode == "+":
                    if type not in head_variables and type not in body_variables:  # noqa: E501
                        potential = False
                        break
            if potential:
                potential_modes.append(index)
        return potential_modes

    def _set_target(self):
        head_variables = {}
        for predicate, *arguments in self._modes:
            if predicate == self._target:
                for _, argument_type in arguments:
                    variable = self._variable_factory.get_new_variable()
                    head_variables.setdefault(
                        argument_type, []
                    ).append(variable)
                break
        return head_variables

    def _get_new_literal(self):
        potential_modes_indexes = self._get_potential_modes_indexes(
            self._head_variables,
            self._body_variables
        )
        mode = self._modes[random.choice(potential_modes_indexes)]
        predicate, *mode_arguments = mode
        arguments = []
        for mode_type, argument_type in mode_arguments:
            if mode_type == "+":
                variables = self._head_variables.get(argument_type, []) + \
                    self._body_variables.get(argument_type, [])
                new_argument = random.choice(variables)
            if mode_type == "-":
                variables = [None] + \
                    self._head_variables.get(argument_type, []) + \
                    self._body_variables.get(argument_type, [])
                new_argument = random.choice(variables)
            if mode_type == "`":
                variables = [None] + \
                    self._body_variables.get(argument_type, [])
                new_argument = random.choice(variables)
            if mode_type == "#":
                constant = random.choice(list(self._constants[argument_type]))
                new_argument = Constant(constant)
            if new_argument is None:
                new_argument = self._variable_factory.get_new_variable()
                self._body_variables.setdefault(
                    argument_type, []
                ).append(new_argument)
            arguments.append(new_argument)
        predicate = Predicate(predicate)
        literal = Literal(predicate, arguments)
        return literal

    def _reset_variables(self):
        self._variable_factory = VariableFactory()
        self._head_variables = self._set_target()
        self._body_variables = {}

    def get_clause(self):
        self._reset_variables()
        literals = []
        literals_set = set()
        for i in range(self._max_literals):
            literal = self._get_new_literal()
            # avoid repititions
            for _ in range(self._max_cycles):
                if str(literal) not in literals_set:
                    break
                literal = self._get_new_literal()
            literals_set.add(str(literal))
            literals.append(literal)
        return literals
