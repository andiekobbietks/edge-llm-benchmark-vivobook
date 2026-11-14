# Hardware profile

**Machine:** ASUS VivoBook X513EA / VivoBook 15 (2021-era)

| Item | Recorded value | Confidence / note |
|---|---|---|
| CPU | Intel Core i5-1135G7 | 4 cores / 8 threads, Tiger Lake |
| iGPU | Intel Iris Xe Graphics | 80 EUs; OpenVINO reports GPU architecture v12.0.0 |
| RAM | 8 GB DDR4 | 7.89 GB usable in one capture |
| OS | Windows 10 Home 22H2 | Build 19045 |
| OpenVINO | 2026.4.1 | Recorded on 3 Nov 2026 |
| Graphics driver | 32.0.101.7092 | Installed after initial GPU allocation failures; 2026-09-03 |
| Previous driver | 30.0.101.1404 | 2022-02-18 |
| Python | 3.12.10 | Recorded experiment |
| Ollama | 0.40.13 | Recorded experiment |

## Earlier device diagnostics

- OpenVINO enumerated `CPU` and `GPU`.
- GPU architecture was reported as `vendor=0x8086 arch=v12.0.0`.
- `DEVICE_MEMORY_STATISTICS` was unsupported by the GPU plugin in the original diagnostic.
- The GPU path repeatedly failed to allocate 8,388,608 bytes for USM Device or USM Host.
- The LFM3-1.7B INT4 OpenVINO experiment loaded in 4.26 seconds but subsequently failed a 6,291,456-byte CPU allocation when approximately 1,105 MB was available.

## Interpretation

These records are historical observations, not a guarantee that the current driver/runtime combination behaves identically. Re-run `scripts/collect_system.py` on the target machine before each new benchmark campaign. Do not publish serial numbers, usernames, device IDs or other unnecessary host identifiers in public results.
