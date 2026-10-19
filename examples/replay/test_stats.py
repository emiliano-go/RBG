from stats import median


def test_odd_length():
    assert median([3, 1, 2]) == 2


def test_even_length():
    assert median([1, 2, 3, 4]) == 2.5
