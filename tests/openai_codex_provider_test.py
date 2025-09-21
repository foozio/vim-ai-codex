import pytest

from providers.openai_codex import OpenAICodexProvider


class DummyUtils:
    def print_debug(self, text, *args):
        pass

    def make_known_error(self, message):
        return RuntimeError(message)

    def load_api_key(self, env_variable, token_file_path="", token_load_fn=""):
        return "test-key"


def _make_messages():
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Say hi"}],
        },
    ]


def test_codex_provider_formats_prompt(monkeypatch):
    utils = DummyUtils()
    provider = OpenAICodexProvider("complete", {}, utils)

    captured = {}

    def fake_request(url, data, options):
        captured["url"] = url
        captured["data"] = data
        captured["options"] = options
        yield {"choices": [{"text": "Hello there"}]}

    monkeypatch.setattr(provider, "_openai_request", fake_request)

    chunks = list(provider.request(_make_messages()))

    assert captured["url"] == "https://api.openai.com/v1/completions"
    assert "Assistant::" in captured["data"]["prompt"]
    assert chunks == [{"type": "assistant", "content": "Hello there"}]


def test_codex_provider_rejects_chat():
    utils = DummyUtils()
    with pytest.raises(RuntimeError):
        OpenAICodexProvider("chat", {}, utils)
