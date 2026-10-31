"""Model-aware local token estimates using tiktoken."""

from typing import Any, Protocol


class Encoding(Protocol):
    name: str

    def encode(self, text: str, *, disallowed_special: tuple[()]) -> list[int]: ...


class TokenizerConfigurationError(RuntimeError):
    """Raised when tiktoken or a requested encoding is unavailable."""


class TiktokenTokenizer:
    """Count text locally while exposing which encoding was selected."""

    def __init__(
        self,
        model: str,
        fallback_encoding: str = "cl100k_base",
        encoder: Encoding | Any | None = None,
    ) -> None:
        self.model = model
        if encoder is not None:
            self._encoder = encoder
            self.used_fallback = False
        else:
            self._encoder, self.used_fallback = _load_encoding(model, fallback_encoding)
        self.encoding_name = self._encoder.name

    def count_text(self, text: str) -> int:
        """Return a local estimate for the token count of a text value."""
        return len(self._encoder.encode(text, disallowed_special=()))


def _load_encoding(model: str, fallback_encoding: str) -> tuple[Encoding, bool]:
    try:
        import tiktoken
    except ImportError as error:
        raise TokenizerConfigurationError(
            "Install tokenization support with `uv sync --extra tokenization`"
        ) from error

    try:
        return tiktoken.encoding_for_model(model), False
    except KeyError:
        try:
            return tiktoken.get_encoding(fallback_encoding), True
        except ValueError as error:
            raise TokenizerConfigurationError(
                f"Unknown fallback encoding: {fallback_encoding}"
            ) from error