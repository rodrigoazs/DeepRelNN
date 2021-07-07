from deeprelnn.fol import Variable


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
