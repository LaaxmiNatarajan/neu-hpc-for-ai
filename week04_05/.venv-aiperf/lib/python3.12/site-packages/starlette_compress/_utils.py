from __future__ import annotations

from functools import lru_cache

from starlette.datastructures import MutableHeaders

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Literal

    from starlette.types import Message

_SUPPORTED_ENCODINGS = frozenset({'br', 'gzip', 'zstd'})
_MUTABLE_RESPONSE_HEADERS = frozenset({
    b'content-encoding',
    b'content-length',
    b'vary',
})


def mutable_response_headers(message: Message) -> MutableHeaders:
    """Return a case-insensitive view, coalescing repeated Vary fields."""
    raw_headers = message['headers']
    vary_values: list[bytes] | None = None
    for index, (name, value) in enumerate(raw_headers):
        lower_name = name if name.islower() else name.lower()
        if name != lower_name and lower_name in _MUTABLE_RESPONSE_HEADERS:
            raw_headers[index] = (lower_name, value)
        if lower_name == b'vary':
            if vary_values is None:
                vary_values = []
            vary_values.append(value)

    headers = MutableHeaders(raw=raw_headers)
    if vary_values is not None and len(vary_values) > 1:
        headers['Vary'] = b', '.join(vary_values).decode('latin-1')
    return headers


def _accepts_encoding_quality(params: list[str]) -> bool:
    for param in params:
        key, _, value = param.strip().partition('=')
        if key.strip().lower() != 'q':
            continue
        try:
            return float(value) > 0
        except ValueError:
            return False
    return True


@lru_cache(maxsize=128)
def parse_accept_encoding(accept_encoding: str) -> frozenset[str]:
    """Parse the accept encoding header and return a set of supported encodings.

    >>> _parse_accept_encoding('br;q=1.0, gzip;q=0.8, *;q=0.1')
    {'br', 'gzip', 'zstd'}
    """
    accepted: set[str] = set()
    rejected: set[str] = set()
    wildcard = False

    for item in accept_encoding.split(','):
        coding, separator, params = item.partition(';')
        coding = coding.strip().lower()
        if not coding:
            continue

        if not separator or _accepts_encoding_quality(params.split(';')):
            if coding == '*':
                wildcard = True
            elif coding in _SUPPORTED_ENCODINGS:
                accepted.add(coding)
        else:
            rejected.add(coding)
            accepted.discard(coding)

    if wildcard:
        accepted.update(_SUPPORTED_ENCODINGS - rejected)

    return frozenset(accepted)


# Based on
# - https://github.com/h5bp/server-configs-nginx/blob/main/h5bp/web_performance/compression.conf#L38
# - https://developers.cloudflare.com/speed/optimization/content/compression/
_compress_content_types: set[str] = {
    'application/atom+xml',
    'application/connect+json',
    'application/connect+proto',
    'application/eot',
    'application/font-sfnt',
    'application/font-woff',
    'application/font',
    'application/geo+json',
    'application/gpx+xml',
    'application/graphql+json',
    'application/javascript-binast',
    'application/javascript',
    'application/json',
    'application/ld+json',
    'application/manifest+json',
    'application/opentype',
    'application/otf',
    'application/proto',
    'application/protobuf',
    'application/rdf+xml',
    'application/rss+xml',
    'application/truetype',
    'application/ttf',
    'application/vnd.api+json',
    'application/vnd.google.protobuf',
    'application/vnd.mapbox-vector-tile',
    'application/vnd.ms-fontobject',
    'application/wasm',
    'application/x-google-protobuf',
    'application/x-httpd-cgi',
    'application/x-javascript',
    'application/x-opentype',
    'application/x-otf',
    'application/x-perl',
    'application/x-protobuf',
    'application/x-ttf',
    'application/x-web-app-manifest+json',
    'application/xhtml+xml',
    'application/xml',
    'font/eot',
    'font/otf',
    'font/ttf',
    'font/x-woff',
    'image/bmp',
    'image/svg+xml',
    'image/vnd.microsoft.icon',
    'image/x-icon',
    'multipart/bag',
    'multipart/mixed',
    'text/cache-manifest',
    'text/calendar',
    'text/css',
    'text/event-stream',
    'text/html',
    'text/javascript',
    'text/js',
    'text/markdown',
    'text/plain',
    'text/richtext',
    'text/vcard',
    'text/vnd.rim.location.xloc',
    'text/vtt',
    'text/x-component',
    'text/x-cross-domain-policy',
    'text/x-java-source',
    'text/x-markdown',
    'text/x-script',
    'text/xml',
}

# Content types that commit early, bypass minimum_size, and flush per ASGI body message.
_streaming_content_types: set[str] = {
    'text/event-stream',
}


def add_compress_type(content_type: str, *, streaming: bool = False) -> None:
    """Add a new content-type to be compressed.

    When ``streaming=True``, also register the type for early commit, minimum_size
    bypass, and per-message compressor flush. A plain re-add does not demote an
    existing streaming membership.
    """
    content_type = content_type.lower()
    _compress_content_types.add(content_type)
    if streaming:
        _streaming_content_types.add(content_type)


def remove_compress_type(content_type: str) -> None:
    """Remove a content-type from being compressed (and from streaming types)."""
    content_type = content_type.lower()
    _compress_content_types.discard(content_type)
    _streaming_content_types.discard(content_type)


def classify_start_message(
    message: Message,
) -> Literal['skip', 'buffered', 'streaming']:
    """Classify whether a response start should be compressed.

    Returns:
        ``skip`` — leave the response untouched (wrong status, range, precompressed,
        or non-compressible content type).
        ``buffered`` — compress with deferred start (ordinary types).
        ``streaming`` — compress with early-commit / per-message flush policy.
    """
    status: int = message.get('status', 200)
    # 1xx informational, 204 No Content, 205 Reset Content, 304 Not Modified
    if status < 200 or status in (204, 205, 304):
        return 'skip'

    content_type: bytes | None = None
    for name, value in message['headers']:
        name = name.lower()
        if name == b'content-encoding':
            for encoding in value.split(b','):
                encoding = encoding.strip()
                if encoding and encoding.lower() != b'identity':
                    return 'skip'
        elif name == b'content-type':
            content_type = value
        elif name == b'content-range':
            return 'skip'

    if content_type is None:
        return 'skip'

    basic_content_type = content_type.split(b';', maxsplit=1)[0].strip()
    try:
        media_type = basic_content_type.decode('ascii').lower()
    except UnicodeDecodeError:
        return 'skip'

    if media_type not in _compress_content_types:
        return 'skip'
    if media_type in _streaming_content_types:
        return 'streaming'
    return 'buffered'
