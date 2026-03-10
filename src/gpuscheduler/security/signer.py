"""
signer.py

Simple HMAC signer for ACPR proof events.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path


DEFAULT_KEY_PATH = Path("state/signing.key")


class HmacSigner:
    def __init__(self, keyPath: Path = DEFAULT_KEY_PATH):
        self.keyPath = Path(keyPath)
        self._key = self._loadOrCreateKey()

    def _loadOrCreateKey(self) -> bytes:
        envKey = os.environ.get("GPUSCHED_SIGNING_KEY")
        if envKey:
            return envKey.encode("utf-8")

        self.keyPath.parent.mkdir(parents=True, exist_ok=True)

        if self.keyPath.exists():
            data = self.keyPath.read_text().strip()
            if data:
                return data.encode("utf-8")

        generated = os.urandom(32).hex()
        self.keyPath.write_text(generated)
        return generated.encode("utf-8")

    def signDigest(self, digestHex: str) -> str:
        return hmac.new(
            self._key,
            digestHex.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verifyDigest(self, digestHex: str, signatureHex: str) -> bool:
        expected = self.signDigest(digestHex)
        return hmac.compare_digest(expected, signatureHex)

