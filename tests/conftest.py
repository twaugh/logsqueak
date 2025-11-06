"""Shared test fixtures for all test modules."""

import pytest


@pytest.fixture(scope="session")
def shared_sentence_transformer():
    """
    Session-scoped SentenceTransformer encoder shared across all tests.

    This fixture loads the embedding model once at the start of the test session
    and reuses it across all tests, significantly reducing test execution time.

    Loading the model takes ~2 seconds, so sharing it across ~20 RAG tests
    saves ~40 seconds of test time.
    """
    from sentence_transformers import SentenceTransformer

    # Load the same model used by PageIndexer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model
