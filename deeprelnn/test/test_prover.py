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
    pos = ["teste(teste2, teste3)."]
    facts = [
        "teste2(teste2, teste3).",
        "teste3(teste2, teste3, teste4).",
        "teste3(teste2, teste3, teste4).",
        "teste3(teste2, teste3, teste4).",
    ]
    bk = Prover(pos, [], facts)
    assert bk.pos["teste"].shape == (1, 200)
    assert bk.pos["teste"].columns[1] == "teste_1"
    assert bk.facts["teste2"].shape == (1, 2)
    assert bk.facts["teste3"].shape == (3, 3)
    assert bk.facts["teste2"].columns[0] == "teste2_0"
    assert bk.facts["teste3"].columns[1] == "teste3_1"


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
    assert result
    result = prover.prove(
        {"A": ["john"], "B": ["maria"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("B")]),
        ],
    )
    assert not result
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("_"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("_"), Variable("B")]),
        ],
    )
    assert result
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("A")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("B")]),
        ],
    )
    assert result
