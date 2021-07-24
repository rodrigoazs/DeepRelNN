from deeprelnn.structure._criterion import MSE, Gini


def test_gini_literal_impurity():
    att = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    assert Gini.literal_impurity(att, true) == 0.4428571428571429


def test_gini_literal_impurity_assert_zero():
    att = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    true = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    assert Gini.literal_impurity(att, true) == 0.0


def test_gini_literal_impurity_assert_almost_zero():
    att = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    true = [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert Gini.literal_impurity(att, true) == 0.12857142857142853


def test_mse_literal_impurity():
    att = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    assert MSE.literal_impurity(att, true) == 0.22142857142857145


def test_mse_literal_impurity_assert_zero():
    att = [1, 1, 1, 0, 0, 0]
    true = [5, 5, 5, 9, 9, 9]
    assert MSE.literal_impurity(att, true) == 0.0


def test_mse_literal_impurity_assert_almost_zero():
    att = [1, 1, 1, 0, 0, 0]
    true = [5, 4.9, 5, 6, 6, 6.1]
    assert MSE.literal_impurity(att, true) == 0.0022222222222222066
