from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from darkness.localdeploy import LocalDeployStructuredClient, StructuredResponseError


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    reason: str


def envelope(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_invalid_response_is_retried_and_schema_validated() -> None:
    payloads = []
    responses = iter([envelope('{"passed":"not-bool"}'), envelope('{"passed":true,"reason":"fixed"}')])

    def send(payload):
        payloads.append(payload)
        return next(responses)

    result = LocalDeployStructuredClient(sender=send).request(
        model="qwen3_6_27b",
        messages=[{"role": "user", "content": "Assess."}],
        response_model=Assessment,
    )
    assert result.passed is True
    assert len(payloads) == 2
    assert payloads[0]["response_format"]["type"] == "json_schema"
    assert "failed schema validation" in payloads[1]["messages"][-1]["content"]


def test_repeated_invalid_response_fails_closed() -> None:
    client = LocalDeployStructuredClient(sender=lambda payload: envelope("not json"))
    with pytest.raises(StructuredResponseError) as error:
        client.request(
            model="qwen3_6_27b",
            messages=[{"role": "user", "content": "Assess."}],
            response_model=Assessment,
            max_attempts=2,
        )
    assert len(error.value.errors) == 2


def test_retry_budget_is_bounded() -> None:
    client = LocalDeployStructuredClient(sender=lambda payload: envelope("{}"))
    with pytest.raises(ValueError, match="between 1 and 3"):
        client.request(
            model="qwen3_6_27b",
            messages=[],
            response_model=Assessment,
            max_attempts=4,
        )
