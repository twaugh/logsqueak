tags:: #testing, #tool
type:: framework
language:: [[Python]]
created:: [[2024-01-21]]

- Pytest is a testing framework for [[Python]]
  - More Pythonic than unittest
  - Rich plugin ecosystem
  - Clear and concise test syntax
- Key features
  id:: 65a2c1f0-2222-4567-89ab-cdef0123456c
  - Simple assert statements
    - No need for self.assertEqual
  - Fixtures for setup/teardown
    - Reusable test components
  - Parametrized tests
    - Run same test with different inputs
  - Detailed failure reports
- Fixtures
  - Function-scoped (default)
  - Class-scoped
  - Module-scoped
  - Session-scoped
    - Share expensive setup across tests
- Plugins
  - pytest-cov for coverage reports
  - pytest-mock for mocking
  - pytest-asyncio for async tests
  - pytest-django for [[Django]] testing
- Markers
  - @pytest.mark.skip
  - @pytest.mark.parametrize
  - @pytest.mark.slow
    - Custom markers for organization
- Configuration
  - pytest.ini or pyproject.toml
  - Set default options
  - Configure coverage thresholds
  - Define custom markers
