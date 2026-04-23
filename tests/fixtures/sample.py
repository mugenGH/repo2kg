"""Sample Python fixture for repo2kg tests."""
import os
import json
from pathlib import Path


def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def calls_other(name: str) -> str:
    result = helper_func(name)
    return f"processed: {result}"


def helper_func(value: str) -> str:
    return value.strip().lower()


class SampleClass:
    """A sample class for testing."""

    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello, {self.name}"

    def process(self, data: list) -> dict:
        result = standalone_function(len(data), 1)
        return {"count": result, "name": self.name}


def recursive_a():
    return recursive_b()


def recursive_b():
    return recursive_a()
