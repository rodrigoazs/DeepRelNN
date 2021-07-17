import pytest

from deeprelnn.factory import ClauseFactory, VariableFactory
from deeprelnn.fol import Variable


def test_variable_factory_initia_variables():
    factory = VariableFactory([Variable("A"), Variable("B")])
    assert factory.variables == set(["A", "B"])

    factory = VariableFactory([Variable("A"), Variable("B"), Variable("Var1")])
    assert factory.variables == set(["A", "B", "Var1"])

    factory = VariableFactory([Variable("B"), Variable("Var1"), Variable("Var2")])
    assert factory.variables == set(["B", "Var1", "Var2"])

    factory = VariableFactory()
    assert factory.variables == set()


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

    factory = VariableFactory()  
    assert factory.get_new_variable() == Variable("A")
    assert factory.get_new_variable() == Variable("B")
    assert factory.get_new_variable() == Variable("C")


def test_clause_factory_potential_modes_indexes():
    modes = [
        "actor(+person).",
        "personlovesgender(+person,#gender).",
        "moviegender(+movie,#gender).",
        "advisedby(+person,`person).",
        "moviegender(-movie,#gender).",
        "actor(-person).",
    ]

    facts = []
    target = "actor"
    factory = ClauseFactory(modes, facts, target)

    head_variables = {
        "person": [Variable("A"), Variable("B")]
    }

    body_variables = {
        "movie": [Variable("C")]
    }

    assert factory._get_potential_modes_indexes(head_variables, body_variables) == [0, 1, 2, 3, 4, 5]
    assert factory._get_potential_modes_indexes({}, body_variables) == [2, 4, 5]
    assert factory._get_potential_modes_indexes(head_variables, {}) == [0, 1, 3, 4, 5]


def test_clause_factory_set_target():
    modes = [
        "actor(+person)",
        "personlovesgender(+person,#gender)",
        "moviegender(+movie,+gender)",
        "advisedby(+person,`person)",
        "moviegender(-movie,#gender)",
        "actor(-person)",
    ]

    facts = []
    factory = ClauseFactory(modes, facts, "advisedby")
    assert factory._head_variables == {"person": [Variable("A"), Variable("B")]}
    factory = ClauseFactory(modes, facts, "actor")
    assert factory._head_variables == {"person": [Variable("A")]}
    factory = ClauseFactory(modes, facts, "moviegender")
    assert factory._head_variables == {"movie": [Variable("A")], "gender": [Variable("B")]}


def test_clause_factory_get_first_literal():
    modes = [
        "actor(+person).",
        "personlovesgender(+person,#gender).",
        "moviegender(+movie,+gender).",
        "advisedby(+person,`person).",
        "moviegender(-movie,#gender).",
        "actor(-person).",
    ]

    facts = [
        "moviegender(movie1,gender1).",
        "moviegender(movie1,gender2).",
    ]

    factory = ClauseFactory(modes, facts, "advisedby")
    # all possibilities
    assert str(factory._get_new_literal()) in [
        'personlovesgender(A, "gender1")',
        'personlovesgender(A, "gender2")',
        'personlovesgender(B, "gender1")',
        'personlovesgender(B, "gender2")',
        'advisedby(B, C)',
        'advisedby(A, C)',
        'actor(A)',
        'actor(B)',
        'actor(C)',
        'moviegender(A, "gender1")',
        'moviegender(A, "gender2")',
        'moviegender(B, "gender1")',
        'moviegender(B, "gender2")',
        'moviegender(C, "gender1")',
        'moviegender(C, "gender2")',
    ]


def test_clause_factory_get_clause():
    modes = [
        "actor(+person).",
        "personlovesgender(+person,#gender).",
        "moviegender(+movie,+gender).",
        "advisedby(+person,`person).",
        "moviegender(-movie,#gender).",
        "actor(-person).",
    ]

    facts = [
        "moviegender(movie1,gender1).",
        "moviegender(movie1,gender2).",
    ]

    factory = ClauseFactory(modes, facts, "advisedby")
    assert len(factory.get_clause()) == 4
    factory = ClauseFactory(modes, facts, "advisedby", max_literals=3)
    assert len(factory.get_clause()) == 3


def test_clause_factory_disallow_recursion():
    modes = [
        "actor(+person).",
        "advisedby(+person,`person).",
    ]
    facts = []
    factory = ClauseFactory(modes, facts, "advisedby", allow_recursion=False)
    # all possibilities
    assert str(factory._get_new_literal()) in [
        "actor(A)",
        "actor(B)",
    ]

    modes = [
        "advisedby(+person,`person)",
    ]
    facts = []
    factory = ClauseFactory(modes, facts, "advisedby", allow_recursion=False)
    # error empty list
    with pytest.raises(IndexError):
        factory._get_new_literal()


def test_clause_factory_imdb_example():
    modes = [
        "workedunder(+person,-person).",
        "workedunder(-person,+person).",
        "female(+person).",
        "actor(+person).",
        "director(+person).",
        "movie(+movie,+person).",
        "movie(+movie,-person).",
        "movie(-movie,+person).",
        "genre(+person,-genre)."
    ]

    facts = []
    factory = ClauseFactory(modes, facts, "workedunder", allow_recursion=False)
    assert len(factory.get_clause()) == 4
    factory = ClauseFactory(modes, facts, "workedunder", max_literals=3)
    assert len(factory.get_clause()) == 3
