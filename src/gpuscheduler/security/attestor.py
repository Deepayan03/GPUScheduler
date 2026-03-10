"""
attestor.py

Attestation provider layer for ACPR.
The MVP ships with a local mock attestor so the full flow can run offline.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict


class MockAttestor:
    def __init__(self, providerName: str = "mock"):
        self.providerName = providerName

    def attest(self, gpuIndex: int) -> Dict[str, Any]:
        ts = time.time()
        deviceId = f"mock-gpu-{gpuIndex}"
        firmwareHash = hashlib.sha256(
            f"fw::{deviceId}".encode("utf-8")
        ).hexdigest()

        return {
            "provider": self.providerName,
            "gpuIndex": gpuIndex,
            "deviceId": deviceId,
            "firmwareHash": firmwareHash,
            "driverVersion": "mock-1.0.0",
            "ccMode": "simulated",
            "timestamp": ts,
        }


def isAttestationCompliant(
    attestation: Dict[str, Any],
    policy: Dict[str, Any],
) -> bool:
    if not policy:
        return True

    requiredProvider = policy.get("requiredProvider")
    if requiredProvider and attestation.get("provider") != requiredProvider:
        return False

    requiredCcMode = policy.get("requiredCcMode")
    if requiredCcMode and attestation.get("ccMode") != requiredCcMode:
        return False

    allowedDeviceIds = policy.get("allowedDeviceIds")
    if allowedDeviceIds:
        if attestation.get("deviceId") not in set(allowedDeviceIds):
            return False

    firmwarePrefix = policy.get("firmwareHashPrefix")
    if firmwarePrefix and not str(attestation.get("firmwareHash", "")).startswith(
        str(firmwarePrefix)
    ):
        return False

    requiredDriverPrefix = policy.get("requiredDriverPrefix")
    if requiredDriverPrefix and not str(attestation.get("driverVersion", "")).startswith(
        str(requiredDriverPrefix)
    ):
        return False

    return True

