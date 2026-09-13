"""
VERDICT Database Layer
Lightweight persistent storage for verification history, claims, evidence, and certificates.
"""

from backend.database.connection import get_db_connection, get_db_context, get_db_path
from backend.database.schema import init_db
from backend.database.repository import (
    save_verification,
    get_verification,
    get_certificate_from_db,
    list_verifications,
    get_verification_count
)

__all__ = [
    "get_db_connection",
    "get_db_context",
    "get_db_path",
    "init_db",
    "save_verification",
    "get_verification",
    "get_certificate_from_db",
    "list_verifications",
    "get_verification_count"
]
