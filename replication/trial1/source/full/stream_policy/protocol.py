"""Messages between the simulator process and the policy server (standard library only).

A message is a 4-byte big-endian header length, a UTF-8 JSON header, then `nbytes` raw payload
bytes (an RGB frame: uint8, row-major H x W x 3, its shape in the header). Replies with an
"error" key carry the server-side traceback.
"""
import json
import struct


def send(stream, header, payload=b""):
    data = json.dumps(dict(header, nbytes=len(payload)), separators=(",", ":"), allow_nan=False).encode()
    stream.write(struct.pack(">I", len(data)) + data)
    if payload:
        stream.write(payload)
    stream.flush()


def _read(stream, n):
    chunks, left = [], n
    while left:
        chunk = stream.read(left)
        if not chunk:
            raise EOFError(f"stream closed with {left} of {n} bytes unread")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def receive(stream):
    """(header, payload), or (None, b"") at a clean end of stream."""
    first = stream.read(4)
    if not first:
        return None, b""
    if len(first) < 4:
        first += _read(stream, 4 - len(first))
    header = json.loads(_read(stream, struct.unpack(">I", first)[0]))
    return header, _read(stream, header.get("nbytes", 0))
