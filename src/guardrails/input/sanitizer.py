# TODO: Implement `InputSanitizer` — cleans and normalises raw user input
#       before it reaches the guardrail validator or the RAG pipeline.
#
# Suggested transformations (implement at minimum the first two):
# 1. Strip leading/trailing whitespace and collapse repeated internal whitespace.
# 2. Truncate messages that exceed a safe maximum length (e.g. 2000 chars)
#    to prevent prompt-stuffing attacks.
# 3. Detect and neutralise common prompt-injection patterns, for example:
#    - "Ignore previous instructions..."
#    - "You are now DAN..."
#    - Role-play override attempts ("pretend you are...", "act as...")
#    Flag these rather than silently dropping them so the caller can log them.
# 4. Optionally normalise unicode (NFKC) to prevent homoglyph attacks.
#
# Interface suggestion:
#   class InputSanitizer:
#       def sanitize(self, text: str) -> tuple[str, bool]:
#           """Returns (cleaned_text, was_modified)."""
