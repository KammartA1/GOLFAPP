"""
Shared database session helper for service modules.

The database.connection.get_session() is a bare generator, not decorated
with @contextmanager. This module wraps it into a proper context manager
that all service modules can import.
"""
from contextlib import contextmanager
from database.connection import get_session as _get_session


@contextmanager
def get_session():
    """
    Yield a SQLAlchemy session that auto-commits on clean exit
    and rolls back on exception.
    """
    gen = _get_session()
    session = next(gen)
    try:
        yield session
        # Normal exit: resume generator past the yield so it commits
        try:
            next(gen)
        except StopIteration:
            pass
    except Exception as exc:
        # Error exit: throw into generator so it rolls back
        try:
            gen.throw(type(exc), exc)
        except StopIteration:
            pass
        raise
