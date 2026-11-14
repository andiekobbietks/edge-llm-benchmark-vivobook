# LittleBit × OpenVINO × Intel: feasibility assessment

**Status:** exploratory research. No native LittleBit-on-OpenVINO execution has been demonstrated by this repository.

## 1. What LittleBit is

SamsungLabs describes LittleBit as sub-1-bit LLM compression through low-rank latent factorisation: dense weight matrices are factorised, latent factors are binarised, and learned scales restore magnitude information. LittleBit-2 adds Internal Latent Rotation with Joint Iterative Quantization (Joint-ITQ) during initialisation.

The upstream README lists OPT, Llama families, Phi-4, Qwen families and Gemma families. Its documented environment recommends Python 3.12, PyTorch 2.8.0 CUDA 12.4 and Transformers 4.51.x. This is a training/evaluation research workflow, not an OpenVINO deployment recipe.

## 2. What the LinkedIn claim does and does not mean

A binary factor operation may be implemented with bitwise comparisons and population counts. Depending on the sign encoding, this can be expressed using XNOR-popcount or XOR plus sign/offset handling. It is not accurate to assume that every transformer operation becomes XOR, or that matrix multiplication disappears from the entire model.

The reported kernel-level speedup must not be interpreted as an end-to-end speedup on an Iris Xe laptop. Attention, normalisation, embeddings, residual paths, KV cache, memory movement, tokenisation and kernel launch overhead remain relevant.

## 3. CUDA-free route

The project will investigate four layers independently:

### A. Existing OpenVINO baseline

Run a known pre-converted OpenVINO GenAI model on `CPU` and `GPU`. This validates the runtime, device selection and benchmark harness independently of LittleBit.

### B. Mathematical primitive

Implement a small, independently written CPU reference for packed binary dot products (XNOR/popcount or equivalent signed-bit arithmetic). Validate it against a floating-point reference. This is a microbenchmark only; it is not the LittleBit architecture and must not be labelled as a LittleBit model.

### C. OpenVINO graph/backend feasibility

Determine whether the required factorised operations can be represented using existing OpenVINO ops and optimised by the CPU/GPU plugins. Record graph compilation, device placement and fallback. If custom operations are required, scope a separate C++/OpenCL plugin or custom kernel prototype. Do not claim GPU execution based solely on requesting `GPU`; verify placement and performance.

### D. Checkpoint/export compatibility

Inspect upstream checkpoint tensor names, shapes, factorisation metadata and scale parameters. Build a clean-room converter only after the format is understood and the relevant licence permits the intended use. Compare outputs numerically against a reference implementation.

## 4. Hardware-specific hypothesis

The VivoBook's Tiger Lake iGPU is an integrated GPU sharing system memory with an 8 GB host. Previous measurements show the tested LFM2.5-1.2B workload was faster on CPU (40.96 tok/s) than Iris Xe GPU (15.28 tok/s). This is a useful baseline and warns against assuming GPU offload will help.

**Hypothesis:** the CPU is the lowest-risk first target for a CUDA-free binary primitive. Iris Xe support is a separate optimisation question, not a prerequisite for proving correctness.

## 5. Acceptance gates

| Gate | Pass condition |
|---|---|
| G0: baseline | Existing OpenVINO INT4 model runs reproducibly on CPU |
| G1: reference correctness | Binary primitive matches a trusted numerical reference within defined tolerance |
| G2: CPU performance | Repeatable timing and memory results beat or justify comparison to a reference implementation |
| G3: OpenVINO representation | Equivalent factorised operation is expressible and numerically validated |
| G4: Iris Xe execution | Evidence confirms GPU kernel placement, no hidden CPU fallback, and stable memory behaviour |
| G5: model-level integration | A compatible checkpoint runs end-to-end with quality and latency measurements |
| G6: deployment | Reproducible packaging, supported model formats, diagnostics, tests and licence review |

A failed gate is a useful result. Record it with the complete command, versions, output and error.

## 6. Licensing and IP

The official [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) README states CC BY-NC 4.0. That is a non-commercial licence. Do not copy upstream source into a commercial product, train or distribute derived artefacts for commercial use, or assume that the research paper grants commercial rights. Obtain legal review and contact the rights holders for a commercial licence where necessary.

This repository's MIT licence covers only original files authored here. It does not cover upstream source, papers, model weights, datasets, or third-party libraries. A clean-room reimplementation still requires an IP/patent and licence review; clean-room development does not automatically resolve patent or other rights.

## 7. Upstream references

- [Official implementation and current instructions](https://github.com/SamsungLabs/LittleBit)
- [OpenVINO Intel GPU plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
- [OpenVINO NNCF](https://github.com/openvinotoolkit/nncf)
