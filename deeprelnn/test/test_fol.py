from deeprelnn.fol import Atom, Constant, Literal, Predicate, Term, Variable


def test_create_term():
    term = Term("Test")
    assert term.name == "Test"


def test_create_variable():
    variable = Variable("A")
    assert variable.name == "A"
    assert variable.is_grounded() is False
    assert str(variable) == "A"


def test_variable_assert_equality():
    variable1 = Variable("A")
    variable2 = Variable("A")
    assert variable1 == variable2
    assert len(set({variable1, variable2}))


def test_create_constant():
    const = Constant("const")
    assert const.name == "const"
    assert const.is_grounded() is True
    assert str(const) == '"const"'
    const = Constant('"const"')
    assert const.name == "const"


def test_constant_assert_equality():
    const1 = Constant("test1")
    const2 = Constant("test1")
    assert const1 == const2
    assert len(set({const1, const2}))


def test_create_predicate():
    pred = Predicate("relation")
    assert pred.name == "relation"
    assert str(pred) == "relation"


def test_predicate_assert_equality():
    pred1 = Predicate("predicate")
    pred2 = Predicate("predicate")
    assert pred1 == pred2
    assert len(set({pred1, pred2}))


def test_create_atom():
    predicate = Predicate("relation")
    arguments = [Variable("A"), Variable("B")]
    atom = Atom(predicate, arguments)
    assert atom.predicate.name == "relation"
    assert atom.arguments == arguments
    assert atom.weight == 1.0


def test_atom_assert_equality():
    predicate = Predicate("relation")
    arguments = [Variable("A"), Variable("B")]
    atom1 = Atom(predicate, arguments)
    atom2 = Atom(predicate, arguments)
    assert atom1 == atom2
    assert len(set({atom1, atom2}))


def test_create_weighted_atom():
    predicate = Predicate("relation")
    arguments = [Variable("A"), Variable("B")]
    atom = Atom(predicate, arguments, 2.3)
    assert atom.predicate.name == "relation"
    assert atom.arguments == arguments
    assert atom.weight == 2.3


def test_create_literal():
    predicate = Predicate("relation")
    arguments = [Variable("A"), Variable("B")]
    literal = Literal(predicate, arguments)
    assert literal.predicate.name == "relation"
    assert literal.arguments == arguments
    assert str(literal) == "relation(A, B)"
    assert repr(literal) == "Literal(Predicate(relation)(Variable(A), Variable(B)))"
    assert literal.weight == 1.0
