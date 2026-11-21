from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_answer_and_sources():
    response = client.post("/ask", json={"question": "What is RAML?"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["source"] == "mulesoft.md"


def test_empty_question_rejected():
    assert client.post("/ask", json={"question": "  "}).status_code == 400
