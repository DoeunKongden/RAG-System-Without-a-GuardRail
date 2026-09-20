import re
import unicodedata
from abc import ABC, abstractmethod


class BaseCleaner(ABC):
    @abstractmethod
    def clean(self, text: str) -> str: ...

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class WikiCleaner(BaseCleaner):
    _CITATION = re.compile(r"\[\d+\]")
    _SECTION_DIVIDER = re.compile(r"={2,}[^=]+={2,}")

    def clean(self, text: str) -> str:
        text = self._CITATION.sub("", text)
        text = self._SECTION_DIVIDER.sub("", text)
        return self._normalize_whitespace(text)


class PDFCleaner(BaseCleaner):
    _HYPHEN_BREAK = re.compile(r"-\n(\w)")
    _PAGE_NUMBER = re.compile(r"\n\s*\d+\s*\n")
    _MULTI_SPACE = re.compile(r" {2,}")
    _LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})

    def clean(self, text: str) -> str:
        text = text.translate(self._LIGATURES)
        text = self._HYPHEN_BREAK.sub(r"\1", text)   # re-join hyphenated line breaks
        text = self._PAGE_NUMBER.sub("\n", text)
        text = self._MULTI_SPACE.sub(" ", text)
        return self._normalize_whitespace(text)


class TextCleaner(BaseCleaner):
    def clean(self, text: str) -> str:
        text = text.lstrip("﻿").replace("\x00", "")   # strip BOM + null bytes
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return self._normalize_whitespace(text)


class WebCleaner(BaseCleaner):
    _HTML_ENTITY = re.compile(r"&[a-zA-Z]{2,6};|&#\d+;")
    _MULTI_SPACE = re.compile(r" {2,}")

    def clean(self, text: str) -> str:
        text = self._HTML_ENTITY.sub(" ", text)
        text = self._MULTI_SPACE.sub(" ", text)
        return self._normalize_whitespace(text)
