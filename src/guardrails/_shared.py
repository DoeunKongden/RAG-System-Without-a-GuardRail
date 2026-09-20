"""Shared parsing utilities used by both input and output LlamaGuard validators."""
from kongden_chatbot.src.models.base import GuardrailDecision
from kongden_chatbot.src.models.guardrail import GuardrailViolation, SafetyCategory

# LlamaGuard3 S-code → SafetyCategory
CODE_TO_CATEGORY: dict[str, SafetyCategory] = {
    "S1": SafetyCategory.VIOLENT_CRIMES,
    "S2": SafetyCategory.NON_VIOLENT_CRIMES,
    "S3": SafetyCategory.SEX_CRIMES,
    "S4": SafetyCategory.CHILD_EXPLOITATION,
    "S5": SafetyCategory.HATE,
    "S6": SafetyCategory.SELF_HARM,
    "S7": SafetyCategory.SEXUAL_CONTENT,
    "S8": SafetyCategory.ELECTIONS,
    "S9": SafetyCategory.SPECIALIZED_ADVICE,
}

UNSAFE_CONTENT_CATEGORIES = """\
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Hate.
S6: Self-Harm.
S7: Sexual Content.
S8: Elections.
S9: Specialized Advice."""


def parse_llamaguard_response(raw: str) -> tuple[GuardrailDecision, list[GuardrailViolation]]:
    """Parse a LlamaGuard response into a decision and violation list.

    Expected format:
        safe
        OR
        unsafe
        S1,S3
    """
    lines = raw.strip().splitlines()
    verdict = lines[0].strip().lower()

    if verdict == "safe":
        return GuardrailDecision.ALLOW, []

    violations: list[GuardrailViolation] = []
    if len(lines) >= 2:
        for code in [c.strip() for c in lines[1].split(",")]:
            category = CODE_TO_CATEGORY.get(code)
            if category:
                violations.append(GuardrailViolation(
                    category=category,
                    confidence=1.0,
                    reason=f"LlamaGuard flagged category {code}",
                ))

    return GuardrailDecision.BLOCK, violations
