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
 

# create ClauseFactory class
# it receives the modes and parses it
# when parsing it create a dictionary of constant types with their values
# create a method that identifies possible modes to use in the next literal
# create a method that randomly generate literals
# create a method that randomly generate clauses


# def test_clause_factory_potential_modes():
#     modes = [
#         "actor(+person)",
#         "personlovesgender(+person,#gender)",
#         "moviegender(+movie,#gender)",
#     ]

#     facts = []
#     target = Literal(Predicate("advisedby"), [Variable("A"), Variable("B")])

#     factory = ClauseFactory(modes, facts, target_arguments_types)

#     assert factory._get_potential_modes() == []
