# -*- coding: utf-8 -*-

import pytest

from bathyinversionvagues.skeleton import fib

__author__ = "Alexia Mondot"
__copyright__ = "Alexia Mondot"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
