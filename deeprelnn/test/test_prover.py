from deeprelnn.fol import Constant, Literal, Predicate, Variable
from deeprelnn.prover.prover import Prover


def test_get_literal():
    prover = Prover([])
    literal_string = "0.5::professor(person407)."
    weight, predicate, arguments = prover._get_literal(literal_string)
    assert weight == 0.5
    assert predicate == "professor"
    assert arguments == ["person407"]
    literal_string = "recursion_advisedby(person265,person168)."
    weight, predicate, arguments = prover._get_literal(literal_string)
    assert weight == 1.0
    assert predicate == "recursion_advisedby"
    assert arguments == ["person265", "person168"]
    literal_string = "0::recursion_advisedby(person265,person168,person99)."
    weight, predicate, arguments = prover._get_literal(literal_string)
    assert weight == 0.0
    assert predicate == "recursion_advisedby"
    assert arguments == ["person265", "person168", "person99"]


def test_background_knowledge():
    facts = [
        "test2(test2, test3).",
        "test3(test2, test3, test4).",
        "5.0::test3(test2, test3, test4).",
        "test3(test2, test3, test4).",
    ]
    bk = Prover(facts)
    assert bk.facts["test2"].shape == (1, 3)
    assert bk.facts["test3"].shape == (3, 4)
    assert bk.facts["test2"].columns[0] == "test2_0"
    assert bk.facts["test3"].columns[1] == "test3_1"
    assert sum(bk.facts["test3"]["weight"].values) == 7.0


def test_prover():
    facts = [
        "2.0::actor(john).",
        "actor(maria).",
        "director(isaac).",
        "3.4::movie(movie1, john).",
        "movie(movie1, isaac).",
    ]

    prover = Prover(facts)
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("B")]),
        ],
    )
    assert result == [2.0, 1.0, 3.4, 1.0]
    result = prover.prove(
        {"A": ["john"], "B": ["maria"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("B")]),
        ],
    )
    assert result == [2.0, 0.0, 0.0, 0.0]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Constant("test")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("D")]),
        ],
    )
    assert result == [2.0, 1.0, 0.0, 0.0]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("D")]),
            Literal(Predicate("movie"), [Variable("C"), Constant("test")]),
        ],
    )
    assert result == [2.0, 1.0, 2.2, 0]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("_"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("_"), Variable("B")]),
        ],
    )
    assert result == [2.0, 1.0, 3.4, 1.0]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("A")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("B")]),
        ],
    )
    assert result == [2.0, 1.0, 3.4, 1.0]
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Constant("pedro")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("A")]),
            Literal(Predicate("movie"), [Constant("movie1"), Variable("B")]),
        ],
    )
    assert result == [0.0, 0.0, 0.0, 0.0]


def test_prover_middle_weight():
    facts = [
        "2.0::actor(john).",
        "actor(maria).",
        "director(isaac).",
        "3.4::movie(movie1, john).",
        "9.0::movie(movie2, john).",
        "movie(movie1, isaac).",
    ]

    prover = Prover(facts)
    result = prover.prove(
        {"A": ["john"], "B": ["isaac"]},
        [
            Literal(Predicate("actor"), [Variable("A")]),
            Literal(Predicate("director"), [Variable("B")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("A")]),
            Literal(Predicate("movie"), [Variable("C"), Variable("B")]),
        ],
    )
    assert result == [2.0, 1.0, 3.4, 1.0]


def test_prover_family_example():
    facts = [
        "male(mrgranger).",
        "male(jamespotter).",
        "male(harrypotter).",
        "male(luciusmalfoy).",
        "male(dracomalfoy).",
        "male(arthurweasley).",
        "male(ronweasley).",
        "male(fredweasley).",
        "male(georgeweasley).",
        "male(hagrid).",
        "male(dumbledore).",
        "male(xenophiliuslovegood).",
        "male(cygnusblack).",
        "siblingof(ronweasley,fredweasley).",
        "siblingof(ronweasley,georgeweasley).",
        "siblingof(ronweasley,ginnyweasley).",
        "siblingof(fredweasley,ronweasley).",
        "siblingof(fredweasley,georgeweasley).",
        "siblingof(fredweasley,ginnyweasley).",
        "siblingof(georgeweasley,ronweasley).",
        "siblingof(georgeweasley,fredweasley).",
        "siblingof(georgeweasley,ginnyweasley).",
        "siblingof(ginnyweasley,ronweasley).",
        "siblingof(ginnyweasley,fredweasley).",
        "siblingof(ginnyweasley,georgeweasley).",
        "childof(mrgranger,hermione).",
        "childof(mrsgranger,hermione).",
        "childof(jamespotter,harrypotter).",
        "childof(lilypotter,harrypotter).",
        "childof(luciusmalfoy,dracomalfoy).",
        "childof(narcissamalfoy,dracomalfoy).",
        "childof(arthurweasley,ronweasley).",
        "childof(mollyweasley,ronweasley).",
        "childof(arthurweasley,fredweasley).",
        "childof(mollyweasley,fredweasley).",
        "childof(arthurweasley,georgeweasley).",
        "childof(mollyweasley,georgeweasley).",
        "childof(arthurweasley,ginnyweasley).",
        "childof(mollyweasley,ginnyweasley).",
        "childof(xenophiliuslovegood,lunalovegood).",
        "childof(cygnusblack,narcissamalfoy).",
    ]

    prover = Prover(facts)
    result = prover.prove(
        {"A": ["harrypotter"], "B": ["mrgranger"]},
        [
            Literal(Predicate("siblingof"), [Variable("B"), Variable("B")]),
            Literal(Predicate("male"), [Variable("A")]),
            Literal(Predicate("male"), [Variable("B")]),
            Literal(Predicate("siblingof"), [Variable("A"), Variable("B")]),
        ],
    )
    assert result == [0.0, 0.0, 0.0, 0.0]
