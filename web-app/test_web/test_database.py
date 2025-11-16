# tests/test_database.py
# pylint: disable=import-error

import types

import database


class FakeMongoClient:
    """Fake MongoClient to avoid connecting to a real MongoDB server."""

    def __init__(self, uri):
        self.uri = uri
        self.closed = False
        self._db = types.SimpleNamespace(name=None)

    def __getitem__(self, name):
        # client[db_name]
        self._db.name = name
        return self._db

    def close(self):
        self.closed = True


def test_get_database_singleton(monkeypatch):
    """get_database() should return the same DB instance on repeated calls."""
    fake_client = FakeMongoClient("mongodb://example")
    monkeypatch.setattr(database, "MongoClient", lambda uri: fake_client)

    # Ensure clean state
    database.close_database()

    db1 = database.get_database()
    db2 = database.get_database()

    assert db1 is db2
    assert db1.name == database.Config.MONGO_DB_NAME


def test_close_database_resets_client(monkeypatch):
    """close_database() should close the client and allow creation of a new one."""
    fake_client = FakeMongoClient("mongodb://example")
    monkeypatch.setattr(database, "MongoClient", lambda uri: fake_client)

    database.close_database()
    db1 = database.get_database()
    assert not fake_client.closed

    # Close client
    database.close_database()
    assert fake_client.closed is True

    # Next call should use a new client
    fake_client2 = FakeMongoClient("mongodb://example2")
    monkeypatch.setattr(database, "MongoClient", lambda uri: fake_client2)
    db2 = database.get_database()

    assert db1 is not db2
    assert db2.name == database.Config.MONGO_DB_NAME
