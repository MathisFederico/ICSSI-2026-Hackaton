"""Shared pytest fixtures and hooks for altendor tests.

The ``live`` marker is declared in the root ``pyproject.toml`` and excluded
from the default run via ``-m 'not live'``. Tests requiring network or
real-API access must wear ``@pytest.mark.live``.
"""
