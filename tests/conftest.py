"""Pytest configuration and shared fixtures."""

from textwrap import dedent

import pytest


@pytest.fixture
def sample_journal_content():
    """Sample journal content for testing."""
    return dedent(
        """\
        - worked on [[Project X]]
          - Made progress on feature A
        - met with team
        - Discovered [[Project X]] deadline slipping to May
          - Original deadline was March
          - Vendor delays caused the slip
        - attended standup meeting
        - Main competitor is [[Product Y]]
          - They use pricing model Z"""
    )


@pytest.fixture
def sample_page_content():
    """Sample page content for testing."""
    return dedent(
        """\
        - ## Timeline
          - Kickoff: January 2025
          - MVP: March 2025 (original)
        - ## Team
          - Alice (PM)
          - Bob (Eng)
        - ## Status
          - In progress"""
    )


@pytest.fixture
def sample_plain_bullets_page():
    """Sample page with plain bullet convention."""
    return dedent(
        """\
        - Timeline
          - Q1 2025
          - Q2 2025
        - Team
          - Alice
          - Bob
        - Status
          - Active"""
    )


@pytest.fixture
def sample_heading_bullets_page():
    """Sample page with heading bullet convention."""
    return dedent(
        """\
        - ## Timeline
          - Q1 2025
          - Q2 2025
        - ## Team
          - Alice
          - Bob
        - ## Status
          - Active"""
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
