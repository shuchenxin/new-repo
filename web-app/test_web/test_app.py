# tests/test_app.py
# pylint: disable=import-error


def test_index_page_with_results(client):
    """The index page should render correctly and display classifications when data exists."""
    resp = client.get("/")
    assert resp.status_code == 200

    html = resp.data.decode("utf-8")

    # Basic UI elements
    assert "Music Classification Dashboard" in html
    assert "Total Songs" in html
    assert "Vocal" in html
    assert "Instrumental" in html

    # Sample data content
    assert "song_vocal.mp3" in html
    assert "song_instr.mp3" in html

    assert "Vocal" in html
    assert "Instrumental" in html


def test_index_page_no_results(monkeypatch):
    """The index page should show an empty-state message when the DB has no results."""
    import app as app_module  # pylint: disable=import-outside-toplevel

    class EmptyCursor:
        def sort(self, *_args, **_kwargs):
            return self

        def limit(self, _n):
            return []

    class EmptyCollection:
        def find(self, *_args, **_kwargs):
            return EmptyCursor()

        def count_documents(self, _filter):
            return 0

    class EmptyDB:
        def __init__(self):
            self.classifications = EmptyCollection()

    def get_empty_db():
        return EmptyDB()

    # Patch get_database() to use an empty DB
    monkeypatch.setattr(app_module, "get_database", get_empty_db)
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "No classification results yet." in html


def test_api_results(client):
    """Test JSON structure returned by /api/results."""
    resp = client.get("/api/results")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0

    first = data["results"][0]
    assert isinstance(first["_id"], str)
    assert "classification" in first
    assert "confidence" in first
    assert "timestamp" in first


def test_api_stats(client):
    """Test /api/stats for correct counting and percentage calculation."""
    resp = client.get("/api/stats")
    assert resp.status_code == 200

    data = resp.get_json()
    total = data["total"]
    vocal = data["vocal"]
    instrumental = data["instrumental"]

    # Total = vocal + instrumental
    assert total == vocal + instrumental

    if total > 0:
        expected_vocal_pct = round(vocal / total * 100, 2)
        expected_instr_pct = round(instrumental / total * 100, 2)
        assert data["vocal_percentage"] == expected_vocal_pct
        assert data["instrumental_percentage"] == expected_instr_pct
    else:
        assert data["vocal_percentage"] == 0
        assert data["instrumental_percentage"] == 0
