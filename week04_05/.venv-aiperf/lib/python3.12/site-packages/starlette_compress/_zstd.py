from __future__ import annotations

# This backend is imported only on Python 3.14+, beyond the checker's 3.11 target.
from compression.zstd import (  # pyright: ignore[reportMissingImports]
    ZstdCompressor,  # pyright: ignore[reportAttributeAccessIssue]
)

from starlette_compress._responder import CompressionResponder

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Iterable

    from starlette.types import ASGIApp


class _ZstdStreamEncoder:
    __slots__ = ('_compressor',)

    def __init__(self, level: int) -> None:
        self._compressor = ZstdCompressor(level)

    def feed(self, data: bytes, flush: bool) -> Iterable[bytes]:
        mode = ZstdCompressor.FLUSH_BLOCK if flush else ZstdCompressor.CONTINUE
        out = self._compressor.compress(data, mode)
        return (out,) if out else ()

    def finish(self) -> Iterable[bytes]:
        out = self._compressor.flush(ZstdCompressor.FLUSH_FRAME)
        return (out,) if out else ()


class ZstdResponder(CompressionResponder):
    __slots__ = ()

    def __init__(self, app: ASGIApp, minimum_size: int, level: int) -> None:
        compressor = ZstdCompressor(level)

        def oneshot(body: bytes) -> bytes:
            return compressor.compress(body, ZstdCompressor.FLUSH_FRAME)

        def create_encoder(content_length: int) -> _ZstdStreamEncoder:
            return _ZstdStreamEncoder(level)

        super().__init__(app, minimum_size, 'zstd', oneshot, create_encoder)
