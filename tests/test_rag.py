from app.rag import Assistant, Retriever, build_prompt, load_chunks
from evaluate import evaluate

retriever = Retriever(load_chunks())


def test_documents_are_chunked_by_section():
    chunks = load_chunks()
    sections = {(c.source, c.section) for c in chunks}
    assert ("kubernetes.md", "Probes") in sections
    assert len(chunks) >= 12


def test_retrieves_correct_section():
    top_chunk, _ = retriever.search("readiness vs liveness probe")[0]
    assert top_chunk.section == "Probes"


def test_unrelated_question_returns_no_sources():
    result = Assistant(retriever, llm=None).ask("xyzzy qwerty")
    assert result["sources"] == []


def test_extractive_mode_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = Assistant(retriever).ask("What is DataWeave?")
    assert result["mode"] == "extractive"
    assert "JSON" in result["answer"]


def test_llm_receives_grounded_prompt():
    captured = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return "fake answer"

    result = Assistant(retriever, llm=fake_llm).ask("What is a Service in Kubernetes?")
    assert result["mode"] == "llm"
    assert result["answer"] == "fake answer"
    assert "ONLY the context" in captured["prompt"]
    assert "stable network address" in captured["prompt"]


def test_prompt_contains_question():
    prompt = build_prompt("Q?", load_chunks()[:1])
    assert "Question: Q?" in prompt


def test_retrieval_quality_meets_threshold():
    metrics = evaluate(k=3)
    assert metrics["hit_rate@3"] >= 0.9
    assert metrics["mrr"] >= 0.75
