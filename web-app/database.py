"""
Database connection module for MongoDB.
"""

from pymongo import MongoClient
from config import Config

_client = None
_db = None


def get_database():
    """
    Get or create database connection.

    Returns:
        Database: MongoDB database instance
    """
    global _client, _db

    if _db is None:
        _client = MongoClient(Config.MONGO_URI)
        _db = _client[Config.MONGO_DB_NAME]

    return _db


def close_database():
    """Close database connection."""
    global _client, _db

    if _client is not None:
        _client.close()
        _client = None
        _db = None
