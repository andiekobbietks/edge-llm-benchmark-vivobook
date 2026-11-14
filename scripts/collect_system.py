#!/usr/bin/env python3
"""Collect a privacy-conscious Windows/OpenVINO hardware and runtime profile."""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception:
        return None

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/system-profile.json")
    args = parser.parse_args()
    profile = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python": sys.version,
        "machine_architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        profile["memory_bytes"] = {"total": vm.total, "available": vm.available, "used": vm.used}
        profile["cpu"] = {"brand": platform.processor(), "physical_cores": psutil.cpu_count(logical=False),
                          "logical_cores": psutil.cpu_count(logical=True)}
    except Exception as exc:
        profile["psutil_error"] = repr(exc)
    try:
        import openvino as ov
        core = ov.Core()
        profile["openvino"] = {"version": ov.__version__, "available_devices": core.available_devices}
        devices = {}
        for device in core.available_devices:
            try:
                devices[device] = {
                    "full_name": core.get_property(device, "FULL_DEVICE_NAME"),
                    "architecture": core.get_property(device, "DEVICE_ARCHITECTURE"),
                }
            except Exception as exc:
                devices[device] = {"property_error": repr(exc)}
        profile["openvino"]["devices"] = devices
    except Exception as exc:
        profile["openvino_error"] = repr(exc)
    profile["gpu_driver_query"] = command_output(["powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate | ConvertTo-Json -Compress"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {output}")
    print(json.dumps(profile, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
