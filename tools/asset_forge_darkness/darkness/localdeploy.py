"""Schema-validating LocalDeploy client with bounded corrective retry."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)
Sender = Callable[[dict[str, Any]], dict[str, Any]]


class StructuredResponseError(RuntimeError):
    def __init__(self, errors: list[str]):
        super().__init__("LocalDeploy did not return a schema-valid response: " + " | ".join(errors))
        self.errors = errors


class LocalDeployStructuredClient(Generic[T]):
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "local",
        timeout_seconds: float = 300,
        sender: Sender | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._sender = sender or self._send_http

    def _send_http(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text.replace("\n", " ")[:1200]
                raise RuntimeError(
                    f"LocalDeploy HTTP {response.status_code}: {detail}"
                ) from exc
            return response.json()

    def request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_attempts: int = 2,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> T:
        if max_attempts < 1 or max_attempts > 3:
            raise ValueError("max_attempts must be between 1 and 3")
        working_messages = [dict(message) for message in messages]
        errors: list[str] = []
        schema = response_model.model_json_schema()

        for attempt in range(1, max_attempts + 1):
            payload = {
                "model": model,
                "messages": working_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            try:
                envelope = self._sender(payload)
                content = envelope["choices"][0]["message"]["content"]
                return response_model.model_validate_json(content)
            except (KeyError, IndexError, TypeError, ValidationError, ValueError) as exc:
                summary = str(exc).replace("\n", " ")[:1200]
                errors.append(f"attempt {attempt}: {summary}")
                if attempt >= max_attempts:
                    break
                prior = str(locals().get("content", ""))[:4000]
                working_messages.extend(
                    [
                        {"role": "assistant", "content": prior},
                        {
                            "role": "user",
                            "content": (
                                "The prior response failed schema validation. Return one corrected JSON object only. "
                                f"Validation error: {summary}"
                            ),
                        },
                    ]
                )
        raise StructuredResponseError(errors)
