import pytest

from deeprelnn.utils import check_random_state


def test_check_random_state():
    assert check_random_state(1)
    assert check_random_state(None)
    with pytest.raises(ValueError):
        assert check_random_state("test")
