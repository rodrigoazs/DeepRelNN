from deeprelnn.fol import (
    Variable,
    Atom,
    Constant,
    HornClause,



    Literal,
    Predicate,
    Term,
)


def test_create_term():
    term = Term("Test")
    assert term.name == "Test"


def test_create_variable():
    variable = Variable("A")
    assert variable.name == "A"
    assert variable.is_grounded() is False
    assert str(variable) == "A"


def test_create_constant():
    const = Constant("const")
    assert const.name == "const"
    assert const.is_grounded() is True
    assert str(const) == '"const"'
    const = Constant('"const"')
    assert const.name == "const"


def test_create_predicate():
    pred = Predicate("relation")
    assert pred.name == "relation"
    assert str(pred) == "relation"


def test_create_atom():
    predicate = Predicate("relation")
    arguments = [Variable("A"), Variable("B")]
    atom = Atom(predicate, arguments)
    assert atom.predicate.name == "relation"
    assert atom.arguments == arguments


def test_create_literal():
    predicate = Predicate("relation")
    arguments = [Variable("A"), Variable("B")]
    literal = Literal(predicate, arguments)
    assert literal.predicate.name == "relation"
    assert literal.arguments == arguments
    assert str(literal) == "relation(A, B)"
    assert repr(literal) == "Literal(Predicate(relation)(Variable(A), Variable(B)))"


def test_create_horn_clause():
    head = Literal(Predicate("test"), [Variable("A")])
    tail = [
        Literal(Predicate("follow"), [Variable("A")]),
        Literal(Predicate("friend"), [Variable("A")]),
    ]
    clause = HornClause(head, tail, 0.5)
    assert clause
