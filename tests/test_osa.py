import pytest
import textdistance
from deep_ml import optimal_string_alignment


@pytest.mark.parametrize(
    "s1, s2",
    [
        ("abc", ""),
        ("d", "c"),
        ("abcdfsafd", "2132sfsdaf31edssf"),
        ("ab", "ba"),
        ("abc", "9cb"),
        ("", "")
    ]
)
def test_optimal_string_alignment(s1, s2):
    osa = textdistance.DamerauLevenshtein()
    assert optimal_string_alignment(s1, s2) == osa.distance(s1, s2)
