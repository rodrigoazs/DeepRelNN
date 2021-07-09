from deeprelnn.fol import Variable
from deeprelnn.parser import get_literal, get_modes


class VariableFactory:
    def __init__(self, initial_variables):
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
    def __init__(self, modes, facts):
        self._modes = self._parse_modes(modes)
        self._constants = self._get_constants(facts)

    def _parse_modes(self, modes):
        return get_modes(modes)

    def _get_constants(self, facts):
        types = {}
        constants = {}
        for mode in self._modes:
            for index, argument in enumerate(mode[1:]):
                if argument[0] == "#":
                    types.setdefault(
                        mode[0], {}
                    ).setdefault(index, argument[1])
        for fact in facts:
            _, predicate, arguments = get_literal(fact)
            if predicate in types:
                for key, value in types.get(predicate).items():
                    constants.setdefault(value, []).append(arguments[key])
        return constants
