import random
import string
from hashlib import sha256
import sqlite3
from contextlib import contextmanager


def stable_hash(s: str) -> str:
    return sha256(s.encode('utf-8')).hexdigest()

def init_cache_db(path: str):
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS execution_cache (
                code_hash TEXT PRIMARY KEY,
                execution_time INTEGER
            )
        """)
        conn.commit()

# Context manager to handle database connections safely
@contextmanager
def open_cache_db(path: str):
    conn = sqlite3.connect(path, timeout=5)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def generate_random_string():
    """Generate a random string of length 10"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))


def print_info(*args):
    """Prints an information message"""
    message = ' '.join(map(str, args))
    print(f"\033[94m[INFO]\t {message}\033[0m")


def print_success(*args):
    """Prints a success message"""
    message = ' '.join(map(str, args))
    print(f"\033[92m[SUCCESS]\t {message}\033[0m")


def print_alert(*args):
    """Prints an alert message"""
    message = ' '.join(map(str, args))
    print(f"\033[93m[ALERT]\t {message}\033[0m")


def print_error(*args):
    """Prints an error message"""
    message = ' '.join(map(str, args))
    print(f"\033[91m[ERROR]\t {message}\033[0m")
