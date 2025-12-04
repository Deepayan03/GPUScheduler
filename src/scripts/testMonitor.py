# scripts/testMonitor.py
from gpuscheduler.daemon.monitor import Monitor, getGpuStatsSnapshot
import time

# 1) quick snapshot
snap = getGpuStatsSnapshot()
print("Snapshot:", snap)

# 2) start background monitor with callback
def cb(s):
    print("CB:", s.get("backend"), "ts:", s.get("timestamp"))

mon = Monitor(pollInterval=2.0, callback=cb)
mon.start()
print("Monitor started — watching for 10 seconds...")
time.sleep(10)
mon.stop()
print("Stopped monitor. Last stats:", mon.getLastStats())