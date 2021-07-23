import pytest

from deeprelnn.fol import Constant
from deeprelnn.parser import get_constants, get_literal, get_modes


def test_get_literal():
    literal_string = "0.5::professor(person407)."
    weight, predicate, arguments = get_literal(literal_string)
    assert weight == 0.5
    assert predicate == "professor"
    assert arguments == ["person407"]
    literal_string = "recursion_advisedby (person265,person168)."
    weight, predicate, arguments = get_literal(literal_string)
    assert weight == 1.0
    assert predicate == "recursion_advisedby"
    assert arguments == ["person265", "person168"]
    literal_string = "0::recursion_advisedby(person265,person168, person99)."
    weight, predicate, arguments = get_literal(literal_string)
    assert weight == 0.0
    assert predicate == "recursion_advisedby"
    assert arguments == ["person265", "person168", "person99"]


@pytest.mark.parametrize("literal", [
    "",
    "ab",
    "(person)",
    "(person).",
    "professor(person407)",
    "actor()",
    "0.2::(person)",
    "ab::(person).",
    ".::(person).",
])
def test_get_literal_assert_raises_error(literal):
    with pytest.raises(ValueError):
        assert get_literal(literal)


def test_get_modes():
    modes = [
        "actor(+person).",
        "actor(-person).",
        "advisedby(+person,-person).",
        "advisedby(-person,+person).",
        "moviegender(+movie, #gender).",
        "moviegender(-movie,+gender).",
        "moviegender( +movie,-gender).",
        "advisedby (+person,  `person).",
        "advisedby(`person,+person).",
    ]

    modes = get_modes(modes)
    assert modes[0] == ["actor", ("+", "person")]
    assert modes[1] == ["actor", ("-", "person")]
    assert modes[2] == ["advisedby", ("+", "person"), ("-", "person")]
    assert modes[3] == ["advisedby", ("-", "person"), ("+", "person")]
    assert modes[4] == ["moviegender", ("+", "movie"), ("#", "gender")]
    assert modes[5] == ["moviegender", ("-", "movie"), ("+", "gender")]
    assert modes[6] == ["moviegender", ("+", "movie"), ("-", "gender")]
    assert modes[7] == ["advisedby", ("+", "person"), ("`", "person")]
    assert modes[8] == ["advisedby", ("`", "person"), ("+", "person")]


@pytest.mark.parametrize("modes", [
    ["actor(&person,%person)."],
    ["actor+person"],
    [""],
    ["(+person,+person)."],
    ["actor"],
    ["actor()"]
])
def test_get_modes_assert_raises_error(modes):
    with pytest.raises(ValueError):
        assert get_modes(modes)


def test_get_constant_types():
    modes = [
        "actor(+person).",
        "personlovesgender(+person,#gender).",
        "moviegender(+movie,#gender).",
    ]

    facts = [
        "personlovesgender(person1, horror).",
        "personlovesgender(person2, scifi).",
        "moviegender(person1, comedy).",
    ]

    modes = get_modes(modes)
    constants = get_constants(modes, facts)
    assert constants["gender"] == set([Constant("horror"), Constant("scifi"), Constant("comedy")])
