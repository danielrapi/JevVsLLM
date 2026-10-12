from jevbench.adapters import OpenAILetterAdapter
from jevbench.schema import Example


class FakeUsage:
    input_tokens = 10
    output_tokens = 1


class FakeResponse:
    id = "resp_test"
    model = "gpt-5.1"
    output_text = "B"
    usage = FakeUsage()
    status = "completed"
    incomplete_details = None


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_openai_letter_adapter_sends_reasoning_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = OpenAILetterAdapter(model="gpt-5.1", reasoning_effort="none")
    fake_client = FakeClient()
    adapter.client = fake_client

    result = adapter.predict(
        Example(
            id="x",
            question="Classify sentiment: great.",
            options=["negative", "positive"],
            answer="B",
        )
    )

    assert result.correct is True
    assert fake_client.responses.kwargs["reasoning"] == {"effort": "none"}
