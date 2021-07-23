import re

from deeprelnn.fol import Constant


def get_literal(literal_string):
    """Parses the literal string.

    Args:
        literal_string (str): A literal string in the format of:
            `predicate(const, const).` or
            `0.5::predicate(const, const).`

    Returns:
        tuple: A tuple containining the literal's weight,
            predicate and arguments.
    """
    literals = re.match(
        "([0-9\.]+)\s*\:\:\s*([a-zA-Z0-9\_]+)\s*\(([a-zA-Z0-9\,\s\_]*)\)\.",
        literal_string
    )
    weight = 1.0
    if literals:
        weight, predicate, arguments = literals.groups()
    else:
        literals = re.match(
            "([a-zA-Z0-9\_]+)\s*\(([a-zA-Z0-9\,\s\_]*)\)\.", literal_string
        )
        if not literals:
            raise ValueError(
                "\"{}\" literal cannot be parsed".format(literal_string)
            )
        predicate, arguments = literals.groups()
    arguments = re.sub("\s", "", arguments).split(",")
    return float(weight), predicate, arguments


def get_modes(modes):
    """Parse the modes strings.

    Args:
        modes (List[str]): A list of modes strings in the
            format of:
            `predicate(-type,+type).`.
            It accepts "+", "-", "#" and "`" modes.

    Returns:
        List[tuple]: A list with tuples of parsed modes
        in the format of `(predicate, [["+", type], ...])`
    """
    parsed_modes = []
    for mode in modes:
        parsed_mode = []
        parsed = re.match(
            "([a-zA-Z0-9\_]+)\s*\(([a-zA-Z0-9\,\_\+\-\`\#\s]*)\)",
            mode
        )
        if not parsed:
            raise ValueError("\"{}\" mode cannot be parsed".format(mode))
        predicate, arguments = parsed.groups()
        arguments = re.sub("\s", "", arguments).split(",")
        parsed_mode.append(predicate)
        for argument in arguments:
            if not len(argument):
                raise ValueError("\"{}\" mode cannot be parsed".format(mode))
            parsed_mode.append((argument[0], argument[1:]))
        parsed_modes.append(parsed_mode)
    return parsed_modes


def get_constants(modes, facts):
    """Given modes and facts it returns a dict with the
    all the constants required in modes.

    Args:
        modes (List[tuple]): Parsed modes.
        facts (List[str]): List of literal strings of the
            facts.

    Returns:
        dict: A dict with a list of constants for each "#"
            mode required.
    """
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
                constants.setdefault(value, set()) \
                    .add(Constant(arguments[key]))
    return constants
