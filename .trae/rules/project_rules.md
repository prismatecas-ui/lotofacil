# Rules for Python System Upgrade Project

## General Principles
- The assistant must execute each requested step **linearly and in order**.
- Do not deviate the focus toward optimizations or unrelated tasks.
- Keep track of every action to simplify future steps.
- Prioritize **correct execution over speed**.

## Error Analysis and Fixing
- Analyze errors **precisely and objectively**, without speculation.
- Always propose a direct and viable solution in Python 3.
- Document inline the causes and suggested fixes.
- Validate the solution with practical examples when applicable.

## Memory and Continuity
- Preserve in memory each decision made during the project.
- Do not repeat resolved steps unless explicitly requested.
- Adjust new tasks based on the history of previous changes.

## Technical Rules
- Main language: **Python 3.x**.
- Core framework: **Flask + SQLAlchemy**.
- Database: **MySQL** with `utf8mb4_unicode_520_ci`.
- Formatting standard: **Black** with `line-length = 88`.
- Use **Pydantic** for data validation when applicable.
- Dependency management via `requirements.txt` or `pyproject.toml`.

## Best Practices
- Do not introduce external dependencies without approval.
- Review SQLAlchemy models before applying migrations.
- Apply migrations with Alembic and review generated files manually.
- Document each module with clear docstrings in English.
- Follow naming convention: `els_` for functions and `els-` for CSS classes.

## Restrictions
- Forbidden to use hardcoded keys or secrets (must use `.env`).
- Do not expose sensitive data in logs or stack traces.
- Do not alter previous steps without notifying the user.
- Do not create partial solutions that rely on external assumptions.

## Example of Correct Execution
1. User requests a fix for an error in a SQLAlchemy model.  
2. Assistant analyzes the model, detects the error, proposes the exact fix.  
3. Suggests the Alembic migration command.  
4. Documents the change and keeps it in memory for future modifications.  
