tags:: #testing, #methodology
type:: practice
created:: [[2024-01-20]]

- Test-Driven Development (TDD) is a software development approach
  - Write tests before implementation code
  - Red-Green-Refactor cycle
- The TDD cycle
  id:: 65a2c1f0-1111-4567-89ab-cdef0123456b
  - Red: Write a failing test
    - Test should fail for the right reason
  - Green: Write minimal code to pass
    - Don't worry about perfection yet
  - Refactor: Clean up the code
    - Improve design without changing behavior
- Benefits
  - Better code design
  - Higher test coverage
  - Documentation through tests
  - Confidence in refactoring
  - Fewer bugs in production
- Common pitfalls
  - Testing implementation details
  - Not refactoring enough
  - Writing too many tests at once
  - Skipping the refactor step
- Tools for [[Python]]
  - [[Pytest]] is the most popular
  - unittest from standard library
  - nose2 as alternative
- Best practices
  - Keep tests fast
  - One assertion per test (guideline)
  - Use descriptive test names
  - Follow AAA pattern: Arrange, Act, Assert
    pattern:: AAA
