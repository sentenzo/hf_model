import pytest


def test__true_is_true(always_true: bool) -> None:
    assert always_true


@pytest.mark.slow
def test__true_is_true_for_real(always_true: bool) -> None:
    from time import sleep

    sleep(1)
    assert always_true
