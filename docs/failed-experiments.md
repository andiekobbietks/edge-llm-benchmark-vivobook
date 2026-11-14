# Failed and inconclusive experiments

This file preserves failures as first-class research observations. A failed run is not evidence that a device or runtime is universally unsupported.

## OpenVINO Iris Xe USM allocation failures

**Observed:** 3 November 2026.

- Initial graphics driver: Intel 30.0.101.1404 (2022-02-18).
- OpenVINO reported the Iris Xe device, but `LLMPipeline("ov_model", "GPU")` failed while allocating 8,388,608 bytes for USM Device.
- A later attempt reported the same 8 MiB allocation size for USM Host.
- `DEVICE_MEMORY_STATISTICS` was unsupported by the GPU plugin during the initial diagnostics.
- Driver was subsequently updated to 32.0.101.7092 (2026-09-03). The historical GPU throughput result of 15.28 tok/s is recorded separately; do not assume the allocation failure and successful run used identical software state without checking the original logs.

**Next experiment:** rerun system capture and CPU/GPU benchmarks under the updated driver; preserve exact stderr and record whether failure occurs at plugin initialisation, model compilation, pipeline load or generation.

## LFM3-1.7B INT4 allocation failure

**Observed:** 3 November 2026.

- Model load completed in 4.26 seconds.
- CPU execution then failed to allocate 6,291,456 bytes while approximately 1,105 MB was reported available.
- The failure is consistent with memory pressure/fragmentation or an allocation constraint, but the root cause was not proven.

**Next experiment:** record process working set, commit charge, available physical memory and pagefile state immediately before model load and at failure. Repeat after a clean reboot with no browser or IDE open.

## Earlier reasoning benchmark

A separate recorded CPU run used `ov.LLMPipeline("ov_model", "CPU")`, `max_new_tokens=1200`, generated 693 tokens in 24.03 seconds, and reported 36.04 tok/s with 1.74 seconds load time. Preserve this as a separate historical run: the exact model identity and generation settings must be recovered from the original script/log before merging it into the LFM2.5 comparative table.

## Local conversion access violation

The LFM2.5-1.2B local `optimum-cli export openvino` conversion attempt exited with Windows access violation `0xC0000005` / `-1073741819` on optimum-intel 2.2.0 and 2.3.0.dev0 with Transformers 5.4.0. Pre-converted OpenVINO Hub artefacts were the successful path in the recorded experiment.

The root cause was suspected to involve hybrid convolution/attention model support and/or the native backend, but was not established conclusively. Do not present that suspicion as a confirmed upstream defect.

## Evidence handling

When rerunning, save:
- full command line and working directory;
- stdout and stderr;
- Python package versions (`pip freeze`);
- driver version and OpenVINO device properties;
- RAM/commit/pagefile snapshots;
- model ID and immutable revision;
- timestamp and whether the run was cold or warm.

Do not overwrite the historical record when newer drivers or runtimes change the outcome.
