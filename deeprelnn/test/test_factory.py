import pytest

from deeprelnn.factory import LiteralFactory, VariableFactory
from deeprelnn.fol import Constant, Variable
from deeprelnn.parser import get_modes


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


def test_literal_factory():
    modes = get_modes([
        "actor(+person).",
    ])

    constants = {
        "gender": [Constant("const1"), Constant("const2")],
    }

    head_variables = {
        "person": [Variable("A")],
    }

    body_variables = {
        "person": [Variable("B")],
    }

    factory = LiteralFactory(modes, constants, head_variables, body_variables)
    assert set(map(str, factory.potential_literals())) == {"actor(A)", "actor(B)"}

    modes = get_modes([
        "actor(+person).",
        "abc(+person,#gender).",
    ])

    factory = LiteralFactory(modes, constants, head_variables, body_variables)
    assert set(map(str, factory.potential_literals())) == {'actor(B)', 'abc(A, "const2")', 'abc(B, "const2")', 'actor(A)', 'abc(B, "const1")', 'abc(A, "const1")'}

    modes = get_modes([
        "advisedby(+person,-person).",
        "advisedby(-person,+person).",
    ])

    head_variables = {
        "person": [Variable("A"), Variable("B")],
    }

    body_variables = {
    }

    factory = LiteralFactory(modes, constants, head_variables, body_variables)
    assert set(map(str, factory.potential_literals())) == {'advisedby(C, B)', 'advisedby(A, A)', 'advisedby(A, B)', 'advisedby(A, C)', 'advisedby(B, C)', 'advisedby(B, A)', 'advisedby(C, A)', 'advisedby(B, B)'}


    modes = get_modes([
        "advisedby(+person,`person).",
        "advisedby(`person,+person).",
    ])

    factory = LiteralFactory(modes, constants, head_variables, body_variables)
    assert set(map(str, factory.potential_literals())) == {'advisedby(A, C)', 'advisedby(C, A)', 'advisedby(C, B)', 'advisedby(B, C)'}


def test_literal_factory_imdb_example():
    modes = get_modes([
        "female(+person).",
        "actor(+person).",
        "director(+person).",
        "movie(+movie,+person).",
        "movie(+movie,-person).",
        "movie(-movie,+person).",
        "genre(+person,-genre)."
    ])

    head_variables = {
        "person": [Variable("A"), Variable("B")],
    }

    body_variables = {
    }

    constants = {}

    factory = LiteralFactory(modes, constants, head_variables, body_variables)
    assert set(map(str, factory.potential_literals())) == {'female(A)', 'movie(C, A)', 'female(B)', 'director(B)', 'movie(C, B)', 'actor(A)', 'director(A)', 'actor(B)', 'genre(B, C)', 'genre(A, C)'}

    head_variables = {
        "person": [Variable("A")],
        "movie": [Variable("B")],
    }

    factory = LiteralFactory(modes, constants, head_variables, body_variables)
    assert set(map(str, factory.potential_literals())) == {'actor(A)', 'movie(C, A)', 'genre(A, C)', 'movie(B, A)', 'female(A)', 'movie(B, C)', 'director(A)'}
