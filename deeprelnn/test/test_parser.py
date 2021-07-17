from deeprelnn.parser import get_constants, get_modes


def test_get_modes():
    modes = [
        "actor(+person).",
        "actor(-person).",
        "advisedby(+person,-person).",
        "advisedby(-person,+person).",
        "moviegender(+movie,#gender).",
        "moviegender(-movie,+gender).",
        "moviegender(+movie,-gender).",
        "advisedby(+person,`person).",
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
    assert constants["gender"] == set(["horror", "scifi", "comedy"])
