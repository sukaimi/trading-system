"""Unified LLM client for DeepSeek, Kimi, and Anthropic.

DeepSeek and Kimi both use OpenAI-compatible chat completion APIs.
Anthropic uses its own Messages API format.
When API keys are not set, returns mock responses for testing.
Fallback chain: DeepSeek -> Gemini Flash -> retry.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Type

import requests
from pydantic import BaseModel

from core.logger import setup_logger

log = setup_logger("trading.llm")

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "kimi": {
        "base_url": "https://api.moonshot.ai/v1",
        "model": "moonshot-v1-auto",
        "env_key": "KIMI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-opus-4-6",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.0-flash",
        "env_key": "GOOGLE_API_KEY",
    },
}


class LLMClient:
    """Unified interface for all LLM providers."""

    def __init__(self, mock_mode: bool | None = None):
        self._mock_mode = mock_mode
        self._timeout = 60
        self._cost_tracker = None

    def set_cost_tracker(self, tracker: Any) -> None:
        """Attach a CostTracker to record per-call costs."""
        self._cost_tracker = tracker

    @property
    def mock_mode(self) -> bool:
        if self._mock_mode is not None:
            return self._mock_mode
        return not any(os.getenv(p["env_key"]) for p in PROVIDERS.values())

    def call_deepseek(
        self,
        prompt: str,
        system_prompt: str = "",
        response_schema: Type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Call DeepSeek V3.2 via OpenAI-compatible API."""
        if self.mock_mode:
            return self._get_mock_response("deepseek", response_schema)

        try:
            raw = self._call_openai_compatible("deepseek", prompt, system_prompt)
            result = self._parse_json_response(raw, response_schema)
            if self._cost_tracker and "error" not in result:
                self._cost_tracker.record("deepseek", "pipeline", system_prompt + "\n" + prompt, raw)
            return result
        except Exception as e:
            log.error("DeepSeek call failed: %s", e)
            return {"error": str(e)}

    def call_kimi(
        self,
        prompt: str,
        system_prompt: str = "",
        response_schema: Type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Call Kimi K2.5 via OpenAI-compatible API."""
        if self.mock_mode:
            return self._get_mock_response("kimi", response_schema)

        try:
            raw = self._call_openai_compatible("kimi", prompt, system_prompt)
            result = self._parse_json_response(raw, response_schema)
            if self._cost_tracker and "error" not in result:
                self._cost_tracker.record("kimi", "pipeline", system_prompt + "\n" + prompt, raw)
            return result
        except Exception as e:
            log.error("Kimi call failed: %s", e)
            return {"error": str(e)}

    def call_anthropic(
        self,
        prompt: str,
        system_prompt: str = "",
        response_schema: Type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Call Claude Opus 4.6 via Anthropic Messages API."""
        if self.mock_mode:
            return self._get_mock_response("anthropic", response_schema)

        api_key = os.getenv(PROVIDERS["anthropic"]["env_key"], "")
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not set"}

        try:
            resp = requests.post(
                f"{PROVIDERS['anthropic']['base_url']}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": PROVIDERS["anthropic"]["model"],
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return {"error": f"Anthropic API returned {resp.status_code}"}

            data = resp.json()
            content = data.get("content", [{}])
            raw_text = content[0].get("text", "{}") if content else "{}"
            result = self._parse_json_response(raw_text, response_schema)
            if self._cost_tracker and "error" not in result:
                self._cost_tracker.record("anthropic", "pipeline", system_prompt + "\n" + prompt, raw_text)
            return result
        except Exception as e:
            log.error("Anthropic call failed: %s", e)
            return {"error": str(e)}

    def call_with_fallback(
        self,
        prompt: str,
        system_prompt: str = "",
        response_schema: Type[BaseModel] | None = None,
        primary: str = "deepseek",
    ) -> dict[str, Any]:
        """Call primary provider, fall back to Gemini Flash on failure."""
        # Try primary
        if primary == "deepseek":
            result = self.call_deepseek(prompt, system_prompt, response_schema)
        elif primary == "kimi":
            result = self.call_kimi(prompt, system_prompt, response_schema)
        else:
            result = self.call_deepseek(prompt, system_prompt, response_schema)

        if "error" not in result:
            return result

        log.warning("Primary %s failed, trying Gemini fallback", primary)

        # Try Gemini fallback
        if not self.mock_mode:
            try:
                raw = self._call_gemini(prompt, system_prompt)
                result = self._parse_json_response(raw, response_schema)
                if "error" not in result:
                    return result
            except Exception as e:
                log.warning("Gemini fallback failed: %s", e)

        # Retry primary once
        log.warning("Gemini failed, retrying %s", primary)
        if primary == "deepseek":
            return self.call_deepseek(prompt, system_prompt, response_schema)
        return self.call_kimi(prompt, system_prompt, response_schema)

    # ── Private methods ──────────────────────────────────────────

    def _call_openai_compatible(
        self, provider: str, prompt: str, system_prompt: str
    ) -> str:
        """Make a chat completion call to an OpenAI-compatible API."""
        config = PROVIDERS[provider]
        api_key = os.getenv(config["env_key"], "")
        if not api_key:
            raise ValueError(f"{config['env_key']} not set")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{config['base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config["model"],
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 4096,
            },
            timeout=self._timeout,
        )

        if resp.status_code != 200:
            raise ValueError(f"{provider} API returned {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str, system_prompt: str) -> str:
        """Call Gemini Flash as fallback."""
        api_key = os.getenv(PROVIDERS["gemini"]["env_key"], "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        resp = requests.post(
            f"{PROVIDERS['gemini']['base_url']}/models/{PROVIDERS['gemini']['model']}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": full_prompt}]}],
            },
            timeout=self._timeout,
        )

        if resp.status_code != 200:
            raise ValueError(f"Gemini API returned {resp.status_code}")

        data = resp.json()
        candidates = data.get("candidates", [{}])
        parts = candidates[0].get("content", {}).get("parts", [{}]) if candidates else [{}]
        return parts[0].get("text", "{}") if parts else "{}"

    def _parse_json_response(
        self, raw_text: str, schema: Type[BaseModel] | None = None
    ) -> dict[str, Any]:
        """Extract JSON from LLM response text.

        Handles markdown code blocks and validates against schema.
        """
        text = raw_text.strip()

        # Strip markdown code blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object or array in the text
            for pattern in [r"\{.*\}", r"\[.*\]"]:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                log.warning("Could not parse JSON from LLM response")
                return {"error": "Invalid JSON response", "raw": text[:500]}

        # Validate against schema if provided
        if schema and isinstance(parsed, dict):
            try:
                validated = schema.model_validate(parsed)
                return validated.model_dump()
            except Exception as e:
                log.warning("Schema validation failed: %s", e)
                return parsed

        return parsed

    def _get_mock_response(
        self, provider: str, schema: Type[BaseModel] | None = None
    ) -> dict[str, Any]:
        """Generate a mock response for testing.

        If a schema is provided, constructs a valid default instance.
        """
        if schema:
            try:
                # Try to create a default instance
                instance = schema.model_construct()
                return instance.model_dump()
            except Exception:
                pass

        # Generic mock responses by provider purpose
        return {
            "type": "mock_response",
            "provider": provider,
            "mock": True,
            "message": f"Mock response from {provider}",
        }
