from structural_analytics.response import response_points


def test_response_points():
    result = response_points(
        [0, 10, 20, 30, 25],
        [0, 1, 2, 3, 4],
    )

    assert result["peak_load"] == 30.0
    assert result["peak_displacement"] == 3.0
    assert "ductility" in result
    assert "initial_stiffness" in result
