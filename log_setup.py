"""Logging for the AI service: one format, and secrets never reach the logs.

- ``configure_logging`` sets the format ``time LEVEL logger: message`` and installs
  ``RedactSecretsFilter`` on the root handlers, so even a careless ``logger.info(...)`` cannot write
  a password, a token or a key: they are replaced by ``[REDACTED]`` before the line is written.
"""

from __future__ import annotations

import logging
import re

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
REDACTED = "[REDACTED]"

# (pattern, replacement). Order matters: the most specific ones first.
SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(postgres(?:ql)?(?:\+\w+)?://[^:/\s]+:)[^@\s]+@"), rf"\1{REDACTED}@"),     # DB URL password
    (re.compile(r"\bBearer\s+[\w\-.~+/]+=*", re.IGNORECASE), f"Bearer {REDACTED}"),          # Authorization
    (re.compile(r"\beyJ[\w-]+\.[\w-]+\.[\w-]+"), REDACTED),                                   # any JWT
    (re.compile(r"\brsa:[A-Za-z0-9+/]{16,}={0,2}"), f"rsa:{REDACTED}"),                       # encrypted password
    (re.compile(r"\bx(?:key|smtp)sib-[\w-]+"), REDACTED),                                    # Brevo keys
    (re.compile(r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}"), REDACTED),                            # bcrypt hash
    (re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|token|authorization)(\s*[=:]\s*)(['\"]?)[^\s,'\"}]+\3"),
     rf"\1\2{REDACTED}"),
]


def redact(text: str) -> str:
    """Replaces every secret pattern found in the text with a redaction marker."""
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactSecretsFilter(logging.Filter):
    """Rewrites every record with its secrets hidden (message, arguments and exception text)."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redacts the message and any traceback before the record is written; never drops it."""
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Sets the log format and adds the secret filter to every root handler (idempotent)."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    for handler in logging.getLogger().handlers:
        if not any(isinstance(existing, RedactSecretsFilter) for existing in handler.filters):
            handler.addFilter(RedactSecretsFilter())


