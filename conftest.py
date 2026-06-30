"""Pytest config: ensure repo root is importable and share an expensive tokenizer fixture."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    from src.config import load_config

    return AutoTokenizer.from_pretrained(load_config()["model"]["base"], use_fast=True)
