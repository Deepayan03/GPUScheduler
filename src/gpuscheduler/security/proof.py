"""
proof.py

Proof-chain ledger utilities for ACPR.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from gpuscheduler.daemon.job import Job
from gpuscheduler.security.signer import HmacSigner


def _canonicalJson(data: Dict[str, Any]) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def hashDict(data: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonicalJson(data).encode("utf-8")).hexdigest()


def hashFile(path: str) -> Optional[str]:
    filePath = Path(path)
    if not filePath.exists() or not filePath.is_file():
        return None

    h = hashlib.sha256()
    with filePath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ProofLedger:
    def __init__(
        self,
        proofDir: Path = Path("state/proofs"),
        signer: Optional[HmacSigner] = None,
    ):
        self.proofDir = Path(proofDir)
        self.proofDir.mkdir(parents=True, exist_ok=True)
        self.signer = signer or HmacSigner()

    def appendEvent(
        self,
        job: Job,
        eventType: str,
        gpuIndex: Optional[int],
        attestation: Optional[Dict[str, Any]] = None,
        checkpointHash: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        previousHash = (
            job.proofChain[-1]["eventHash"] if job.proofChain else None
        )

        eventBase = {
            "index": len(job.proofChain),
            "eventType": eventType,
            "timestamp": time.time(),
            "jobId": job.id,
            "gpuIndex": gpuIndex,
            "attestationHash": hashDict(attestation) if attestation else None,
            "checkpointHash": checkpointHash,
            "previousHash": previousHash,
            "extra": extra or {},
        }

        eventHash = hashDict(eventBase)
        signature = self.signer.signDigest(eventHash)

        event = dict(eventBase)
        event["eventHash"] = eventHash
        event["signature"] = signature

        job.proofChain.append(event)
        job.proofStatus = "verified"

        self.writeJobArtifact(job)
        return event

    def verifyJobChain(self, job: Job) -> bool:
        previousHash = None

        for idx, event in enumerate(job.proofChain):
            eventBase = {
                "index": event.get("index"),
                "eventType": event.get("eventType"),
                "timestamp": event.get("timestamp"),
                "jobId": event.get("jobId"),
                "gpuIndex": event.get("gpuIndex"),
                "attestationHash": event.get("attestationHash"),
                "checkpointHash": event.get("checkpointHash"),
                "previousHash": event.get("previousHash"),
                "extra": event.get("extra", {}),
            }

            if eventBase["index"] != idx:
                return False

            if eventBase["previousHash"] != previousHash:
                return False

            expectedHash = hashDict(eventBase)
            if expectedHash != event.get("eventHash"):
                return False

            if not self.signer.verifyDigest(
                expectedHash,
                str(event.get("signature", "")),
            ):
                return False

            previousHash = expectedHash

        return True

    def writeJobArtifact(self, job: Job) -> None:
        payload = {
            "jobId": job.id,
            "proofStatus": job.proofStatus,
            "trustPolicy": job.trustPolicy,
            "chain": job.proofChain,
        }

        outPath = self.proofDir / f"{job.id}.json"
        with outPath.open("w") as f:
            json.dump(payload, f, indent=2)

