from __future__ import annotations

import sys
from operator import index

from starlette.datastructures import MutableHeaders

from starlette_compress._identity import IdentityResponder
from starlette_compress._utils import (
    add_compress_type,
    parse_accept_encoding,
    remove_compress_type,
)

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Iterable

    from starlette.types import ASGIApp, Receive, Scope, Send


def _validate_level(name: str, value: int, maximum: int) -> int:
    try:
        level = index(value)
    except TypeError as error:
        raise TypeError(
            f'{name} must be an integer, got {type(value).__name__}'
        ) from error
    if not 0 <= level <= maximum:
        raise ValueError(f'{name} must be between 0 and {maximum}, got {level}')
    return level


__version__ = '2.0.0'


class CompressMiddleware:
    __slots__ = (
        '_brotli',
        '_gzip',
        '_identity',
        '_remove_accept_encoding',
        '_zstd',
        'app',
    )

    def __init__(
        self,
        app: ASGIApp,
        *,
        minimum_size: int = 500,
        zstd: bool = True,
        zstd_level: int = 4,
        brotli: bool = True,
        brotli_quality: int = 4,
        gzip: bool = True,
        gzip_level: int = 4,
        remove_accept_encoding: bool = False,
    ) -> None:
        """Compression middleware supporting multiple algorithms.

        Positive Accept-Encoding qualities make methods eligible, then server
        priority selects Zstandard, Brotli, Gzip, or no compression (identity).
        Relative quality values do not reorder that priority. Missing/unsupported
        encodings fall back to identity, without a 406 response.

        :param app: ASGI application to wrap.
        :param minimum_size: Minimum ordinary one-shot body size in bytes.
            Multi-message responses begin compression without buffering to this
            threshold. Streaming content types (e.g. ``text/event-stream``) also
            bypass it for one-shot bodies and flush each non-empty continuing
            body message.
        :param zstd: Enable Zstandard compression.
        :param zstd_level: Zstandard compression level. Valid values are all negative integers (faster) to 22 (best).
        :param brotli: Enable Brotli compression.
        :param brotli_quality: Brotli quality level, 0 (fastest) to 11 (best). Checked when enabled.
        :param gzip: Enable Gzip compression.
        :param gzip_level: Gzip compression level, 0 (fastest) to 9 (best). Checked when enabled.
        :param remove_accept_encoding: Remove request Accept-Encoding after reading it.

        :raises TypeError: An enabled Brotli quality or Gzip level is not an integer.
        :raises ValueError: An enabled Brotli quality or Gzip level is outside its valid range.
        """
        if brotli:
            brotli_quality = _validate_level('brotli_quality', brotli_quality, 11)
        if gzip:
            gzip_level = _validate_level('gzip_level', gzip_level, 9)

        self.app = app
        self._identity = IdentityResponder(app, minimum_size)
        self._remove_accept_encoding = remove_accept_encoding

        if zstd:
            if sys.version_info >= (3, 14):
                from starlette_compress._zstd import ZstdResponder
            else:
                from starlette_compress._zstd_legacy import ZstdResponder

            self._zstd = ZstdResponder(app, minimum_size, zstd_level)
        else:
            self._zstd = None

        if brotli:
            from starlette_compress._brotli import BrotliResponder

            self._brotli = BrotliResponder(app, minimum_size, brotli_quality)
        else:
            self._brotli = None

        if gzip:
            from starlette_compress._gzip import GZipResponder

            self._gzip = GZipResponder(app, minimum_size, gzip_level)
        else:
            self._gzip = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        raw_headers: Iterable[tuple[bytes, bytes]] = scope['headers']
        if not isinstance(raw_headers, list):
            raw_headers = scope['headers'] = list(raw_headers)

        accept_encoding = [
            value.decode('latin-1')
            for name, value in raw_headers
            if name == b'accept-encoding'
        ]

        if accept_encoding:
            if self._remove_accept_encoding:
                headers = MutableHeaders(scope=scope)
                del headers['Accept-Encoding']

            accept_encodings = parse_accept_encoding(
                ','.join(accept_encoding)
                if len(accept_encoding) > 1
                else accept_encoding[0]
            )
            if self._zstd is not None and 'zstd' in accept_encodings:
                return await self._zstd(scope, receive, send)
            if self._brotli is not None and 'br' in accept_encodings:
                return await self._brotli(scope, receive, send)
            if self._gzip is not None and 'gzip' in accept_encodings:
                return await self._gzip(scope, receive, send)

        return await self._identity(scope, receive, send)


__all__ = (
    'CompressMiddleware',
    'add_compress_type',
    'remove_compress_type',
)
