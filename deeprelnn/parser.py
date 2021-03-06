import re


def get_literal(literal_string):
    literals = re.match(
        "([0-9\.]*)\s*\:\:\s*([a-zA-Z0-9\_]*)\(([a-zA-Z0-9\,\s\_]*)\)\.",
        literal_string
    )
    weight = 1.0
    if literals:
        weight, predicate, arguments = literals.groups()
    else:
        literals = re.match(
            "([a-zA-Z0-9\_]*)\(([a-zA-Z0-9\,\s\_]*)\)\.", literal_string
        )
        predicate, arguments = literals.groups()
    arguments = re.sub("\s", "", arguments).split(",")
    return float(weight), predicate, arguments


def get_modes(modes):
    parsed_modes = []
    for mode in modes:
        parsed_mode = []
        parsed = re.match(
            "([a-zA-Z0-9\_]*)\(([a-zA-Z0-9\,\_\+\-\`\#\s]*)\)",
            mode
        )
        predicate, arguments = parsed.groups()
        arguments = re.sub("\s", "", arguments).split(",")
        parsed_mode.append(predicate)
        for argument in arguments:
            parsed_mode.append((argument[0], argument[1:]))
        parsed_modes.append(parsed_mode)
    return parsed_modes


def get_constants(modes, facts):
    types = {}
    constants = {}
    for mode in modes:
        for index, argument in enumerate(mode[1:]):
            if argument[0] == "#":
                types.setdefault(
                    mode[0], {}
                ).setdefault(index, argument[1])
    for fact in facts:
        _, predicate, arguments = get_literal(fact)
        if predicate in types:
            for key, value in types.get(predicate).items():
                constants.setdefault(value, set()).add(arguments[key])
    return constants
