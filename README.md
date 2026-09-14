# DiffusionGemma 26B A4B — Tensor Atlas v4

Offline, single-file WebGL2 architecture atlas for **google/diffusiongemma-26B-A4B-it**, built by
retargeting the nisten *Tensor Atlas v4* engine (`LLMViz-DeepSeek-V4.1-Flash`, MIT © 2026 netsin)
with the `tensor-atlas-v4-retarget` skill: the shell (topbar, view tabs, instanced WebGL2
renderer + Canvas2D fallback, inspector, ledger, safetensors audit, modals, CSS) is kept whole;
only the model half is replaced.

Open `index.html` in a browser. No server, no CDN, no build step at runtime.

## What is in the page

| | |
|---|---|
| Blocks | 30 (`qwen`-style hybrid: 25 sliding-window + 5 global) |
| Attention | 16 heads; 8 KV heads × head_dim 256 (sliding) / 2 global KV heads × head_dim 512 (global, **no v_proj stored**) |
| Experts | 128 routed per block, top-8, beside an always-on 2112-wide dense MLP |
| Vision | 27-block tower (16×16 patches, head_dim 72, 4304-wide MLP) + `embed_vision` [2816, 1152] |
| Diffusion | self-conditioning module (2816 → 2112 → 2816), 256-token canvas, ≤48 denoising steps |
| Embedding | 262,144 × 2,816, **tied** (so the checkpoint has no `lm_head`) |
| Context | 262,144 positions, sliding window 1,024 |

## Measured bytes (not estimates)

Every figure was read from the safetensors headers over HTTP Range — no weights downloaded —
then summed per tensor, with per-expert tensors folded into the fused banks the BF16
checkpoint stores and scale tensors folded into their owner.

| Mode | Repository | Shards | Tensors | Payload |
|---|---|---|---|---|
| BF16 | `google/diffusiongemma-26B-A4B-it` | 11 | 1,047 logical | **51.6476 GB** |
| FP8-dynamic | `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` | 1 | 24,232 | **27.1977 GB** |
| NVFP4 | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | 2 | 47,067 | **18.8181 GB** |

`build.py` refuses to write the page unless *page total == measured payload* for all three modes
(8,866,192,664 / … exact integers, tolerance 0), and `node --check` runs on the extracted module.

Where the 51.6 GB goes: routed experts **45.68 GB (88.4%)**, tied embedding 1.48 GB, attention
1.06 GB, vision tower 1.09 GB, shared dense MLPs 0.36 GB, self-conditioning 35.7 MB.

## Files

| file | role |
|---|---|
| `index.html` | the deliverable — self-contained, offline, ~270 KB |
| `build.py` | retarget builder: fold/normalise three measurements → data region → panel rewrites → code patches → gates |
| `src.html` | upstream engine, unmodified (MIT, © 2026 netsin) |
| `measure.py` | safetensors header measurement over HTTP Range (`HF_MEASURE_DIR=/tmp/dg-measure python3 measure.py <repos…>`) |
| `hf-bf16.json` / `hf-fp8.json` / `hf-nvfp4.json` | per-tensor measurement output |
| `hf-config.json` / `hf-config-fp8.json` / `hf-config-nvfp4.json` | published configs, embedded in the config modal |
| `hf-tree-fp8.json`, `README-model-card.md` | provenance |
| `LICENSE` | upstream engine licence (MIT) |

Rebuild: `python3 build.py` (add `--data-only` to emit just the generated data half).

## Method notes

* **Folding.** `experts.N.gate_proj.weight_scale` and friends are mapped onto
  `experts.gate_up_proj`; a tensor that maps to nothing aborts the build.
* **Archetypes.** Layer signatures are grouped by shapes and verified byte-identical, giving
  25 sliding + 5 global block templates; the vision tower is one template counted ×27.
* **Panel rewrites** are content-anchored whole-line replacements (line numbers shift, anchors do
  not), and each must match exactly once or the build stops.
* **Residue gates** assert the original model's vocabulary is gone from the page body and that the
  new model's vocabulary is present.

Apache-2.0 model (Gemma 4 licence), MIT engine. No weight data is bundled.
