import re
import unicodedata
from dataclasses import dataclass, field

MAX_INPUT_LENGTH = 2000

_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)\b', re.IGNORECASE), "ignore_instructions"),
    (re.compile(r'\bforget\s+(all\s+)?(previous|prior|your)\s+(instructions?|training|guidelines?)\b', re.IGNORECASE), "forget_instructions"),
    (re.compile(r'\bdisregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|guidelines?)\b', re.IGNORECASE), "disregard_instructions"),
    (re.compile(r'\boverride\s+(your\s+)?(instructions?|rules?|guidelines?)\b', re.IGNORECASE), "override_instructions"),
    (re.compile(r'\bdo\s+not\s+follow\s+(your\s+)?(instructions?|rules?|guidelines?)\b', re.IGNORECASE), "override_instructions"),
    (re.compile(r'\byou\s+are\s+now\s+\w+', re.IGNORECASE), "persona_override"),
    (re.compile(r'\bpretend\s+(you\s+are|to\s+be)\b', re.IGNORECASE), "pretend"),
    (re.compile(r'\bact\s+as\s+(a\s+|an\s+)?\w+', re.IGNORECASE), "act_as"),
    (re.compile(r'\b(you\s+are\s+)?DAN\b'), "dan_jailbreak"),
    (re.compile(r'\bjailbreak\b', re.IGNORECASE), "jailbreak"),
    (re.compile(r'\bsystem\s+prompt\b', re.IGNORECASE), "system_prompt_reference"),
]


@dataclass
class SanitizationResult:
    cleaned_text: str
    was_modified: bool
    was_truncated: bool = False
    injection_flags: list[str] = field(default_factory=list)

    @property
    def has_injection_attempt(self) -> bool:
        return len(self.injection_flags) > 0


class InputSanitizer:
    def __init__(self, max_length: int = MAX_INPUT_LENGTH, normalize_unicode: bool = True):
        self.max_length = max_length
        self.normalize_unicode = normalize_unicode

    def sanitize(self, text: str) -> tuple[str, bool]:
        """Returns (cleaned_text, was_modified)."""
        result = self.sanitize_full(text)
        return result.cleaned_text, result.was_modified

    def sanitize_full(self, text: str) -> SanitizationResult:
        """Returns a SanitizationResult with full details including injection flags."""
        original = text

        # 1. NFKC unicode normalisation — collapses homoglyphs (e.g. ｉgnore → ignore)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 2. Strip outer whitespace and collapse internal runs
        text = text.strip()
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 3. Truncate to prevent prompt-stuffing
        was_truncated = False
        if len(text) > self.max_length:
            text = text[:self.max_length]
            was_truncated = True

        # 4. Detect prompt-injection patterns — flag, do not silently drop
        injection_flags: list[str] = []
        for pattern, flag in _INJECTION_PATTERNS:
            if pattern.search(text) and flag not in injection_flags:
                injection_flags.append(flag)

        return SanitizationResult(
            cleaned_text=text,
            was_modified=text != original,
            was_truncated=was_truncated,
            injection_flags=injection_flags,
        )
