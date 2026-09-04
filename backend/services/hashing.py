import hashlib
from typing import Union


def compute_sha256(content: Union[bytes, str]) -> str:
    """
    Computes a canonical SHA-256 hexadecimal hash for raw bytes or string content.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(content)
    return hasher.hexdigest()


def compute_file_sha256(file_path: str) -> str:
    """
    Computes SHA-256 of a file on disk using buffered streaming for memory efficiency.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
