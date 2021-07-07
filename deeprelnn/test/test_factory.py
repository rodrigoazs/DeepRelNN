from deeprelnn.factory import VariableFactory
from deeprelnn.fol import Variable


def test_variable_factory_initia_variables():
    factory = VariableFactory([Variable("A"), Variable("B")])
    assert factory.variables == set(["A", "B"])

    factory = VariableFactory([Variable("A"), Variable("B"), Variable("Var1")])
    assert factory.variables == set(["A", "B", "Var1"])

    factory = VariableFactory([Variable("B"), Variable("Var1"), Variable("Var2")])
    assert factory.variables == set(["B", "Var1", "Var2"])


def test_variable_factory_get_new_variable():
    factory = VariableFactory([Variable("A"), Variable("B")])
    assert factory.get_new_variable() == Variable("C")

    factory = VariableFactory([Variable("A"), Variable("B"), Variable("Var1")])
    assert factory.get_new_variable() == Variable("C")

    factory = VariableFactory([Variable("A"), Variable("B"), Variable("C"), Variable("Var1")])
    assert factory.get_new_variable() == Variable("D")

    factory = VariableFactory([Variable(chr(i)) for i in range(ord("A"), ord("Z"))])
    assert factory.get_new_variable() == Variable("Z")

    factory = VariableFactory([Variable(chr(i)) for i in range(ord("A"), ord("Z")+1)])
    assert factory.get_new_variable() == Variable("Var1")

    factory = VariableFactory([Variable(chr(i)) for i in range(ord("A"), ord("Z")+1)] + [Variable("Var1")])
    assert factory.get_new_variable() == Variable("Var2")

    factory = VariableFactory([Variable("B"), Variable("Var3")])
    assert factory.get_new_variable() == Variable("A")
    assert factory.get_new_variable() == Variable("C")
    assert factory.get_new_variable() == Variable("D")
 
