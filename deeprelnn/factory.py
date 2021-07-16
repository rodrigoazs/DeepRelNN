from deeprelnn.fol import Variable
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
    def __init__(self, modes, facts, target):
        self._modes = get_modes(modes)
        self._constants = get_constants(self._modes, facts)
        self._target = target
        self._variable_factory = VariableFactory()
        self._head_variables = self._set_target()
        self._body_variables = {}

    def _get_potential_modes_indexes(self, head_variables, body_variables):
        potential_modes = []
        for index, mode in enumerate(self._modes):
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
        for mode in self._modes:
            if mode[0] == self._target:
                arguments = mode[1:]
                for mode, type in arguments:
                    variable = self._variable_factory.get_new_variable()
                    head_variables.setdefault(type, []).append(variable)
                break
        return head_variables

    def get_new_literal(self):
        pass
