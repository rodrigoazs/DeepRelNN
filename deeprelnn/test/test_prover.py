from deeprelnn.fol import Constant, Literal, Predicate, Variable
from deeprelnn.prover.prover import Prover


def test_get_literal():
    prover = Prover([], [], [])
    literal_string = "professor(person407)."
    predicate, arguments = prover._get_literal(literal_string)
    assert predicate == "professor"
    assert arguments == ["person407"]
    literal_string = "recursion_advisedby(person265,person168)."
    predicate, arguments = prover._get_literal(literal_string)
    assert predicate == "recursion_advisedby"
    assert arguments == ["person265", "person168"]


def test_background_knowledge():
    pos = ["test(test2, test3)."]
    facts = [
        "test2(test2, test3).",
        "test3(test2, test3, test4).",
        "test3(test2, test3, test4).",
        "test3(test2, test3, test4).",
    ]
    bk = Prover(pos, [], facts)
    assert bk.pos["test"].shape == (1, 2)
    assert bk.pos["test"].columns[1] == "test_1"
    assert bk.facts["test2"].shape == (1, 2)
    assert bk.facts["test3"].shape == (3, 3)
    assert bk.facts["test2"].columns[0] == "test2_0"
    assert bk.facts["test3"].columns[1] == "test3_1"


def test_prover():
    facts = [
        "actor(john).",
        "actor(maria).",
        "director(isaac).",
        "movie(movie1, john).",
        "movie(movie1, isaac).",
    ]

    prover = Prover([], [], facts)
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("B")]),
        ],
    )
    assert result == [1, 1, 1, 1]
    result = prover.prove(
        {"A": ["john"], "B": ["maria"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("B")]),
        ],
    )
    assert result == [1, 0, 0, 0]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Constant("test")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("D")]),
        ],
    )
    assert result == [1, 1, 0, 0]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("D")]),
            Literal(Predicate("movie"), [Variable("C"), Constant("test")]),
        ],
    )
    assert result == [1, 1, 1, 0]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("_"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("_"), Variable("B")]),
        ],
    )
    assert result == [1, 1, 1, 1]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("A")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("B")]),
        ],
    )
    assert result == [1, 1, 1, 1]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Constant("pedro")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("A")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("B")]),
        ],
    )
    assert result == [0, 0, 0, 0]
