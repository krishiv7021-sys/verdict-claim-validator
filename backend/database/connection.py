import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/verdict.db"


def get_db_path(custom_path: Optional[str] = None) -> str:
    """
    Resolves the SQLite database path from:
    1. Explicit function argument
    2. DATABASE_PATH environment variable
    3. DATABASE_URL environment variable (handles sqlite:/// format)
    4. Safe default: data/verdict.db
    """
    if custom_path:
        raw_path = custom_path
    else:
        raw_path = os.getenv("DATABASE_PATH") or os.getenv("DATABASE_URL") or DEFAULT_DB_PATH

    # Strip sqlite prefix if provided via DATABASE_URL
    if raw_path.startswith("sqlite:///"):
        raw_path = raw_path[len("sqlite:///"):]
    elif raw_path.startswith("sqlite://"):
        raw_path = raw_path[len("sqlite://"):]

    return raw_path


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Establishes and returns an optimized SQLite connection.
    Applies foreign keys, WAL journal mode (for file DBs), and row factory.
    """
    resolved_path = get_db_path(db_path)

    if resolved_path != ":memory:":
        db_file = Path(resolved_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved_path, timeout=15.0)
    conn.row_factory = sqlite3.Row

    # Performance and integrity pragmas
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")

    if resolved_path != ":memory:":
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        except sqlite3.DatabaseError as e:
            logger.debug(f"Could not set WAL mode: {e}")

    return conn


@contextmanager
def get_db_context(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Transactional context manager for database operations.
    Automatically commits on normal exit and rolls back on exception.
    """
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_db_health(db_path: Optional[str] = None) -> bool:
    """
    Verifies that the database is reachable and operational.
    """
    try:
        with get_db_context(db_path) as conn:
            cursor = conn.execute("SELECT 1;")
            row = cursor.fetchone()
            return row is not None and row[0] == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
