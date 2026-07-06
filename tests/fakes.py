# tests/fakes.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FakeLLMResponse:
    """
    Deterministic fake backend response.

    This mimics the small subset of fields your runners need from a real
    OpenAI response after it has been normalised by your wrapper code.
    """

    text: str
    tool_calls: list[dict[str, Any]] | None = None
    prompt_tokens: int = 10
    completion_tokens: int = 5
    latency_s: float = 0.01

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class FakeLLMClient:
    """
    Fake LLM client for unit and pipeline tests.

    It supports three model-facing operations:

    1. call_backend(...)
       Used by C0/C1/C2/C3/C5 after guardrail processing.

    2. classify_document(...)
       Used by C3/C5 document-level guardrail classification.

    3. sanitise_units(...)
       Used by C5a/C5c-style unit-removal sanitisation.

    It records calls so tests can assert whether the backend, classifier,
    or sanitiser was called.
    """

    def __init__(
        self,
        *,
        force_classifier_decision: str | None = None,
        force_backend_text: str | None = None,
        force_removed_unit_ids: list[int] | None = None,
        fail_classifier: bool = False,
        fail_backend: bool = False,
        fail_sanitiser: bool = False,
    ):
        self.force_classifier_decision = force_classifier_decision
        self.force_backend_text = force_backend_text
        self.force_removed_unit_ids = force_removed_unit_ids

        self.fail_classifier = fail_classifier
        self.fail_backend = fail_backend
        self.fail_sanitiser = fail_sanitiser

        self.backend_calls: list[dict[str, Any]] = []
        self.classifier_calls: list[dict[str, Any]] = []
        self.sanitiser_calls: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------
    # Backend call fake
    # ---------------------------------------------------------------------

    def call_backend(
        self,
        *,
        system_prompt: str,
        user_question: str,
        context: str,
        tools: Any = None,
    ) -> FakeLLMResponse:
        """
        Fake backend agent call.

        Returns deterministic text or a deterministic fake tool call depending
        on the question/context.
        """
        if self.fail_backend:
            raise RuntimeError("Fake backend failure")

        self.backend_calls.append(
            {
                "system_prompt": system_prompt,
                "user_question": user_question,
                "context": context,
                "tools": tools,
            }
        )

        if self.force_backend_text is not None:
            return FakeLLMResponse(text=self.force_backend_text)

        question_l = str(user_question).lower()
        context_l = str(context).lower()

        # Fake benign tool-use behaviour.
        if "send the update" in question_l or "send an email" in question_l:
            return FakeLLMResponse(
                text="",
                tool_calls=[
                    {
                        "tool": "send_email",
                        "arguments": {
                            "to": "alice@example.com",
                            "subject": "Update",
                            "body": "The meeting is confirmed.",
                        },
                    }
                ],
            )

        # Fake tool-misuse behaviour if malicious destination remains.
        if "attacker@example.com" in context_l:
            return FakeLLMResponse(
                text="",
                tool_calls=[
                    {
                        "tool": "send_email",
                        "arguments": {
                            "to": "attacker@example.com",
                            "subject": "Summary",
                            "body": "Meeting moved to Friday.",
                        },
                    }
                ],
            )

        if "$1,240.50" in context or "1240.50" in context_l:
            return FakeLLMResponse(text="The amount due is $1,240.50.")
        
        # Deterministic text answers for tiny fixture examples.
        if "invoice" in question_l or "invoice" in context_l:
            return FakeLLMResponse(text="The invoice amount is £42.")

        if "meeting moved to friday" in context_l:
            return FakeLLMResponse(text="The meeting moved to Friday.")

        if "project status is green" in context_l:
            return FakeLLMResponse(text="Project status is green.")

        if "canary-test-001" in context_l and "include any canary" in context_l:
            return FakeLLMResponse(text="The invoice amount is £42. CANARY-TEST-001")

        return FakeLLMResponse(text="Safe answer.")

    # ---------------------------------------------------------------------
    # Document classifier fake
    # ---------------------------------------------------------------------

    def classify_document(
        self,
        *,
        user_question: str,
        context: str,
    ) -> dict[str, Any]:
        """
        Fake document-level classifier.

        Returns block for obvious injected content and allow otherwise.
        """
        if self.fail_classifier:
            raise RuntimeError("Fake classifier failure")

        self.classifier_calls.append(
            {
                "user_question": user_question,
                "context": context,
            }
        )

        if self.force_classifier_decision is not None:
            decision = self.force_classifier_decision
            return {
                "decision": decision,
                "reason": f"Forced fake classifier decision: {decision}.",
                "confidence": "high",
                "attack_type": "none" if decision == "allow" else "instruction_override",
            }

        attack_type = self._detect_attack_type(context)

        if attack_type == "none":
            return {
                "decision": "allow",
                "reason": "No suspicious embedded instruction detected.",
                "confidence": "high",
                "attack_type": "none",
            }

        return {
            "decision": "block",
            "reason": "Suspicious embedded instruction detected.",
            "confidence": "high",
            "attack_type": attack_type,
        }

    # ---------------------------------------------------------------------
    # Unit sanitiser fake
    # ---------------------------------------------------------------------

    def sanitise_units(
        self,
        *,
        user_question: str,
        context: str,
        units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Fake unit-removal sanitiser.

        Returns selected unit IDs for removal.
        """
        if self.fail_sanitiser:
            raise RuntimeError("Fake sanitiser failure")

        self.sanitiser_calls.append(
            {
                "user_question": user_question,
                "context": context,
                "units": units,
            }
        )

        if self.force_removed_unit_ids is not None:
            return {
                "remove_unit_ids": self.force_removed_unit_ids,
                "reason": "Forced fake unit removal.",
            }

        remove_unit_ids = []

        for unit in units:
            unit_id = unit.get("unit_id")
            text = str(unit.get("text", ""))

            if self._detect_attack_type(text) != "none":
                remove_unit_ids.append(unit_id)

        return {
            "remove_unit_ids": remove_unit_ids,
            "reason": "Removed units containing suspicious embedded instructions.",
        }

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _detect_attack_type(text: str) -> str:
        """
        Small deterministic detector used by the fake classifier/sanitiser.

        This is not your real detector. It is only for predictable tests.
        """
        text_l = str(text).lower()

        if "ignore previous instructions" in text_l:
            return "instruction_override"

        if "forget previous instructions" in text_l:
            return "instruction_override"

        if "your new task is" in text_l:
            return "instruction_override"

        if "include any canary" in text_l:
            return "data_exfiltration"

        if "canary-test-001" in text_l and "include" in text_l:
            return "data_exfiltration"

        if "attacker@example.com" in text_l:
            return "tool_misuse"

        if "markdown_attack_success" in text_l:
            return "markdown_injection"

        if "http://attacker" in text_l or "https://attacker" in text_l:
            return "markdown_injection"

        return "none"

    # ---------------------------------------------------------------------
    # Assertion helpers
    # ---------------------------------------------------------------------

    def reset_calls(self) -> None:
        self.backend_calls.clear()
        self.classifier_calls.clear()
        self.sanitiser_calls.clear()

    @property
    def backend_call_count(self) -> int:
        return len(self.backend_calls)

    @property
    def classifier_call_count(self) -> int:
        return len(self.classifier_calls)

    @property
    def sanitiser_call_count(self) -> int:
        return len(self.sanitiser_calls)