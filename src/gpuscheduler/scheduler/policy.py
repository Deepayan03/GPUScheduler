"""
policy.py

Hybrid scheduling policy engine.

Supports:
- Adaptive GPU utilization analysis
- Moving average tracking
- Spike detection
- Cooldown enforcement
- Static fallback thresholds
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional


class SchedulerPolicy:
    def __init__(
        self,
        staticUtilThreshold: float = 60.0,
        staticMemThreshold: float = 80.0,
        historyWindow: int = 5,
        spikeDelta: float = 25.0,
        cooldownSeconds: float = 5.0,
    ):
        self.staticUtilThreshold = staticUtilThreshold
        self.staticMemThreshold = staticMemThreshold
        self.historyWindow = historyWindow
        self.spikeDelta = spikeDelta
        self.cooldownSeconds = cooldownSeconds

        self._utilHistory: Dict[int, List[float]] = {}
        self._cooldownUntil: Dict[int, float] = {}

    # ----------------------------------------------------
    # History Tracking
    # ----------------------------------------------------

    def updateMetrics(self, gpuIndex: int, utilPercent: float) -> None:
        hist = self._utilHistory.setdefault(gpuIndex, [])
        hist.append(utilPercent)

        if len(hist) > self.historyWindow:
            hist.pop(0)

    def _movingAverage(self, gpuIndex: int) -> Optional[float]:
        hist = self._utilHistory.get(gpuIndex)
        if not hist:
            return None
        return sum(hist) / len(hist)

    # ----------------------------------------------------
    # Spike Detection
    # ----------------------------------------------------

    def _detectSpike(self, gpuIndex: int) -> bool:
        hist = self._utilHistory.get(gpuIndex)
        if not hist or len(hist) < 2:
            return False

        return abs(hist[-1] - hist[-2]) > self.spikeDelta

    # ----------------------------------------------------
    # Cooldown Management
    # ----------------------------------------------------

    def _isCoolingDown(self, gpuIndex: int) -> bool:
        until = self._cooldownUntil.get(gpuIndex)
        if not until:
            return False
        return time.time() < until

    def _triggerCooldown(self, gpuIndex: int) -> None:
        self._cooldownUntil[gpuIndex] = time.time() + self.cooldownSeconds

    # ----------------------------------------------------
    # Decision Logic
    # ----------------------------------------------------

    def canScheduleOnGpu(
        self,
        gpuIndex: int,
        currentUtil: float,
        currentMemUtil: Optional[float] = None,
    ) -> bool:
        """
        Hybrid decision:
        1. If cooling down → reject
        2. If spike detected → cooldown + reject
        3. Adaptive average check
        4. Fallback to static threshold
        """

        self.updateMetrics(gpuIndex, currentUtil)

        if self._isCoolingDown(gpuIndex):
            return False

        if self._detectSpike(gpuIndex):
            self._triggerCooldown(gpuIndex)
            return False

        avg = self._movingAverage(gpuIndex)

        # Adaptive rule
        if avg is not None and avg < self.staticUtilThreshold:
            if currentMemUtil is None:
                return True
            if currentMemUtil < self.staticMemThreshold:
                return True

        # Fallback static
        if currentUtil < self.staticUtilThreshold:
            if currentMemUtil is None:
                return True
            if currentMemUtil < self.staticMemThreshold:
                return True

        return False

    # ----------------------------------------------------
    # Preemption Decision
    # ----------------------------------------------------

    def shouldPreempt(
        self,
        gpuIndex: int,
        currentUtil: float,
        jobPriority: int,
        incomingPriority: int,
    ) -> bool:
        """
        Decide if currently running job should be preempted.
        """

        # Only preempt if new job is higher priority
        if incomingPriority >= jobPriority:
            return False

        # If GPU heavily utilized, avoid preemption
        if currentUtil > 90:
            return False

        return True
    
# Backward compatibility alias
SchedulingPolicy = SchedulerPolicy