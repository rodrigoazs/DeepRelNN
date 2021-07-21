import pytest

from deeprelnn.structure.learner import LearnerClassifier


def test_learner_validate_target():
    modes = [
        "test(+var,-cnt).",
        "test(-var,+cnt).",
        "test2(+cnt,-var).",
    ]
    target = "test"
    learner = LearnerClassifier(modes, target)
    assert learner._validate_target() == None


def test_learner_validate_target_assert_raise_exception_no_target_modes():
    modes = [
        "test2(+cnt,-var).",
    ]
    target = "test"
    learner = LearnerClassifier(modes, target)
    with pytest.raises(ValueError):
        assert learner._validate_target()


def test_learner_validate_modes_assert_raise_exception():
    modes = [
        "test(+var,-cnt).",
        "test(-var,+tre).",
        "test2(+cnt,-var).",
    ]
    target = "test"
    learner = LearnerClassifier(modes, target)
    with pytest.raises(ValueError):
        assert learner._validate_modes()
