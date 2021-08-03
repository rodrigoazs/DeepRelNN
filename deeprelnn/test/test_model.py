from deeprelnn.model import DeepRelNN


def test_deeprelnn_model():
    background = [
        "male(+name).",
        "childof(+name,+name).",
        "siblingof(+name,-name).",
        "father(+name,+name).",
    ]

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

    samples = [
        "0.0::father(harrypotter,mrgranger).",
        "0.0::father(harrypotter,mrsgranger).",
        "0.0::father(georgeweasley,xenophiliuslovegood).",
        "0.0::father(luciusmalfoy,xenophiliuslovegood).",
        "0.0::father(harrypotter,hagrid).",
        "0.0::father(ginnyweasley,dracomalfoy).",
        "0.0::father(hagrid,dracomalfoy).",
        "0.0::father(hagrid,dumbledore).",
        "0.0::father(lunalovegood,dumbledore).",
        "0.0::father(hedwig,narcissamalfoy).",
        "0.0::father(hedwig,lunalovegood).",
        "0.0::father(ronweasley,hedwig).",
        "0.0::father(mollyweasley,cygnusblack).",
        "0.0::father(arthurweasley,mollyweasley).",
        "0.0::father(georgeweasley,fredweasley).",
        "0.0::father(fredweasley,georgeweasley).",
        "0.0::father(ronweasley,georgeweasley).",
        "0.0::father(ronweasley,hermione).",
        "0.0::father(dracomalfoy,narcissamalfoy).",
        "0.0::father(hermione,mrsgranger).",
        "0.0::father(ginnyweasley,mollyweasley).",
        "father(harrypotter,jamespotter).",
        "father(dracomalfoy,luciusmalfoy).",
        "father(ginnyweasley,arthurweasley).",
        "father(ronweasley,arthurweasley).",
        "father(fredweasley,arthurweasley).",
    ]

    model = DeepRelNN(
        background=background,
        target="father",
        number_of_clauses=5,
        allow_recursion=False,
        epochs=1,
    )

    model.fit(samples, facts)
    assert len(model.clauses_) > 1
    
    pred = model.predict_proba(samples, facts)
    assert len(pred) == len(samples)
