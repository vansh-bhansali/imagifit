# Imagifit — Project Notes

Living document for this project. Issues/fixes are a log of what's already been solved (don't redo this work). "What to Build" is the active roadmap. "Context Dump" is raw notes/ideas — unstructured is fine.

---

## Environment

- Local dev machine: MacBook (M3 Pro, Apple Silicon, MPS backend — no CUDA)
- Conda env: `catvton`, Python 3.10, at `/opt/homebrew/Caskroom/miniconda/base/envs/catvton`
  (there's also an unrelated empty `/opt/miniconda3` — ignore it, always activate via the Caskroom path)
- Activate with: `source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate catvton`
- Project code: `CatVTON/` (cloned from [zhengchong/CatVTON](https://github.com/Zheng-Chong/CatVTON))
- Run: `python -u app.py --output_dir=output --mixed_precision=fp16 --width 384 --height 512`
  - `-u` for unbuffered output — without it, redirected stdout doesn't flush until the process exits, making live log-tailing useless.
  - 384×512 resolution chosen over the 768×1024 default as a safer starting point for MPS memory.
- Serves Gradio UI at `http://localhost:7860`

---

## Issues Fixed

### 1. Dependency conflicts in `requirements.txt` (not caught by the original setup guide)
- **Problem**: unpinned `diffusers` (git+main) pulled a version requiring `huggingface_hub>=1.23`, incompatible with the pinned `transformers`/`accelerate`. Unpinned `peft>=0.17.0` pulled a version needing a newer `accelerate` API than pinned `accelerate==0.31.0` provides.
- **Fix**: pinned `diffusers==0.31.0` and `peft==0.11.1` in `requirements.txt`.

### 2. Hardcoded `device='cuda'` throughout `app.py`
- **Problem**: script assumed CUDA everywhere; MPS has no CUDA driver, so it would crash on any Apple Silicon Mac.
- **Fix**: added device auto-detection (`mps` → `cuda` → `cpu` fallback) at the top of `app.py`, replaced all three hardcoded `device='cuda'` references, and used `pipeline.unet.set_attention_slice("auto")` for memory savings (the guide's suggested `pipeline.enable_attention_slicing()` doesn't exist on this custom pipeline class).

### 3. `DensePose` (detectron2) ignored the passed-in device, defaulted to CUDA
- **Problem**: `model/DensePose/__init__.py`'s `setup_config()` built a detectron2 `cfg` via `get_cfg()` but never set `cfg.MODEL.DEVICE` — it silently defaulted to `"cuda"`. Crash: `AssertionError: Torch not compiled with CUDA enabled`. This only surfaces once you actually get far enough to instantiate `AutoMasker` (i.e. after model downloads finish), so it looked like the app was "stuck downloading" when really it had already crashed.
- **Root constraint**: detectron2's custom ops (ROIAlign, NMS, etc.) are compiled C++/CUDA extensions with **no MPS kernels** — so even setting `cfg.MODEL.DEVICE = "mps"` would just move the crash to inference time.
- **Fix**: in `model/DensePose/__init__.py`, force `self.device = "cpu"` when the requested device is `"mps"` (comment explains why), and explicitly set `cfg.MODEL.DEVICE = self.device` before `cfg.freeze()`. DensePose is a one-time pose-estimation preprocessing step, not the main generation — running it on CPU costs some extra time but doesn't block MPS use for the actual diffusion pipeline. (`SCHP`, used right after DensePose, already respected the passed-in device correctly — no fix needed there.)

### 4. `gradio_client` schema parser crash → root page returned HTTP 500
- **Problem**: `gradio_client/utils.py`'s `get_type()` and `_json_schema_to_python_type()` both assume every JSON-schema node is a `dict`. One of the app's components produces a schema with a bare `"additionalProperties": true` (valid JSON Schema, means "any"), and passing that literal `True` into these functions crashed with `TypeError: argument of type 'bool' is not iterable` / `APIInfoParseError: Cannot parse schema True`. This ran on every page load (`gradio/routes.py`'s root handler calls `api_info()` unconditionally to embed API docs in the page — `show_api=False` on `.launch()` does **not** skip this).
- **This is an upstream gradio_client bug**, present in `gradio_client==1.3.0` regardless of the `gradio` version — upgrading `gradio` alone (tried 4.41.0 → 4.44.1) did not fix it, since `gradio_client==1.3.0` is pinned exactly by both.
- **Fix**: runtime monkeypatch in `app.py` (near the top, after imports) wrapping both functions to return `"Any"` when given a non-dict/bool schema, then delegate to the originals otherwise. Chosen over editing installed package files because it's visible, versioned, and survives reinstalls.

### 5. `fastapi`/`starlette` version drift → root page returned HTTP 500 (different bug, same symptom)
- **Problem**: `requirements.txt` never pinned `fastapi` or `starlette`, so pip installed the latest available (`fastapi==0.139.2`, `starlette==1.3.1` at time of install). Starlette's `Jinja2Templates.TemplateResponse` changed signature from `(name, context)` to `(request, name, context)` at some point before 1.0. Gradio's internal code (written for the old signature) calls `templates.TemplateResponse(template, {...})` — under the new signature this silently maps `template` (a string) → `request` param and the context `dict` → `name` param. Jinja then tries to use that dict as part of a cache key → `TypeError: unhashable type: 'dict'`.
- **Fix**: downgraded `fastapi==0.115.0` (contemporaneous with gradio 4.x), which pulled a compatible `starlette==0.38.6` (pre-1.0, old signature) as a transitive dependency.

### 6. `SCHP` (human parsing model) crashed on first real generation request — MPS adaptive pooling
- **Problem**: unlike the startup-time bugs above, this only surfaces when you actually submit an image. `model/SCHP/networks/AugmentCE2P.py`'s context-encoding module (a PSP-style block) runs `AdaptiveAvgPool2d` at four fixed output scales (1×1, 2×2, 3×3, 6×6) on the feature map. MPS's adaptive pooling implementation requires the input size to be **evenly divisible** by the output size — CPU and CUDA don't have this restriction and handle arbitrary sizes. Crash: `RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes` ([pytorch/pytorch#96056](https://github.com/pytorch/pytorch/issues/96056)).
- **Why it wasn't caught earlier**: `SCHP` doesn't hardcode `'cuda'` anywhere (it correctly forwards whatever `device` it's given), so the earlier device-hardcoding sweep didn't flag it. Not hardcoding a bad device isn't the same as being MPS-compatible — its architecture hits a real MPS backend gap regardless of what device string is passed in.
- **Fix**: same CPU-fallback pattern as the DensePose fix (#3) — in `model/SCHP/__init__.py`, force `self.device = "cpu"` when the requested device is `"mps"`, and fixed a bug this introduced where `.to(device)` (the raw param) was still used instead of `.to(self.device)` for moving the model. SCHP runs on CPU now; the main diffusion pipeline still runs on MPS.
- **Takeaway**: "doesn't hardcode a device" and "works on MPS" are different claims — MPS has real operator-level gaps (adaptive pooling here, detectron2's custom ops for DensePose earlier) that only show up when that specific code path actually executes, not at import/init time. If another MPS op error shows up later in a *different* submodule, check for the same class of issue before assuming it's something else.

### 7. ~17GB RAM usage during generation (out of 18GB total)
- **Problem**: generation nearly exhausted system memory. Not a model-size issue — weights are fp16 and CatVTON is already the smallest capable try-on model (~900M params; IDM-VTON/OOTDiffusion need 3-6× more). The memory went to: inference-time activations, the safety-checker model, and above all MPS's caching allocator, which holds freed memory indefinitely so RSS grows across generations and never comes back down.
- **Fix** (no model change):
  - `skip_safety_check=True` on pipeline init in `app.py` — drops the safety-checker CLIP model (~1GB). Local dev only; revisit before exposing publicly.
  - `pipeline.vae.enable_slicing()` + `enable_tiling()` — VAE decodes in chunks, cutting peak activation memory for slightly slower decode.
  - `torch.mps.empty_cache()` after each generation in `submit_function` — releases the MPS allocator cache so memory returns to baseline between requests.
  - Launch with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4` — caps MPS allocations at ~60% of RAM so a single generation can't starve the OS.
- **If memory is still tight**: drop resolution below 384×512 (e.g. `--width 256 --height 384`) — quality degrades but memory scales roughly with pixel count. Swapping to a "smaller model" is not a real option in this space.

### 8. "Images Do Not Match" Error during SD inpainting
- **Problem**: The app crashed with `ValueError: images do not match` at `Image.composite(result_image, person_image, mask_gray)`. The root cause is a size mismatch: the `result_image` (output from SD Inpainting pipeline VAE) or the `mask_gray` (after VAE processing/blur) can be slightly different from the original `person_image` dimensions. `Image.composite` requires all three images to have the exact same size.
- **Fix**: Added explicit `resize()` calls to both `result_image` and `mask_gray` to force them to match `person_image.size` using `Image.LANCZOS` resampling right before the composite step.

### 9. API / Codebase Overengineering and Flask API flaws
- **Problem**: The app used a convoluted Gradio-queue bridge (polling loops) for Flask to run jobs, which caused race conditions and infinite hangs. Flask also returned local file paths (e.g., `/Users/.../result.png`) which the frontend couldn't use.
- **Fix**: 
  - Eliminated the Gradio queue bridging entirely; Flask now directly calls the synchronous `submit_function()`.
  - Replaced `time.time()` based Job IDs with `uuid.uuid4()` to avoid 1-second resolution collisions.
  - Flask `/process` endpoint now reads the saved result image and returns a base64 encoded string (`data:image/png;base64,...`) so the web frontend can directly render the image.
  - Cleaned up unused CLI arguments (`--repaint`, `--allow_tf32`, `--local_rank`), an orphaned `received_images` folder at the project root, and removed an obsolete `gradio_client` monkeypatch. **(Correction: the monkeypatch was NOT obsolete — see #10.)**

### 10. Regression: removing the `gradio_client` monkeypatch (#9) re-broke the Gradio root page
- **Problem**: the cleanup in #9 removed the bool-schema monkeypatch from #4 as "obsolete". It wasn't — the upstream bug in `gradio_client==1.3.0` is still present (it's pinned exactly by gradio 4.x), and every load of the Gradio UI 500'd again with `TypeError: argument of type 'bool' is not iterable`. The Flask `/process` endpoint was unaffected (separate server), which can make the app look "half working".
- **Fix**: restored the monkeypatch in `app.py` (wrapping `get_type` and `_json_schema_to_python_type` to return `"Any"` for bool schema nodes), verified against the exact failing schema shape (`additionalProperties: true`).
- **Takeaway**: this patch must stay until `gradio`/`gradio_client` are upgraded to a version with the upstream fix — do not remove it as dead code again.

### 11. Implemented Asynchronous Webhook Architecture with Static Hosting
- **Problem**: The frontend faced timeout issues waiting for the long AI generation process to complete. Sending the massive base64 image over Localtunnel proxy also triggered `408 Request Timeout` limits on the proxy layer and Express body parser limits.
- **Fix**: Re-architected the app to use an asynchronous two-way webhook model with static file serving:
  - Added a `POST /register` endpoint for the frontend to perform a handshake and register its own `localtunnel` URL.
  - Added a `GET /images/<filename>` endpoint to publicly host generated images statically from the `received_images` folder.
  - Modified `POST /process` to immediately return `HTTP 202 Accepted` and spawn a background thread for image generation.
  - The background worker actively sends a tiny `POST` request back to the registered webhook URL containing just the filename (`generatedImageName`), avoiding all base64 size limits.
  - *Update*: Increased the webhook delivery retry logic from 3 attempts to 10 attempts (with 2-second delays) to provide more resilience against transient network drops.

### Debugging notes / gotchas hit along the way
- Background process output is fully buffered when redirected to a file — always launch with `python -u` or you won't see logs until the process exits.
- When a background process silently dies, `ps aux | grep` returning nothing is the tell — don't assume "still downloading" just because the last known state was mid-download.
- `tail -f` on a file that gets truncated (`>` redirect on relaunch) picks up from the new content correctly, but if you have multiple old `Monitor`/watch processes still tailing the same path, you'll get duplicate/stale-looking notifications that are actually just re-reads of the same file — always stop old monitors before starting new ones for the same log.
- A "server started" log line (`Running on local URL: ...`) does **not** mean the app actually works — Gradio can log that and still 500 on every request. Always verify with an actual HTTP request, not just the log.

---

## What to Build

*(open roadmap — fill in / reorder as needed)*

- [ ] Decide the actual deployment architecture (see note below — the client/Macbook/friend's-laptop diagram had unresolved design issues)
- [x] Wire up the web frontend to actually call this local CatVTON instance (Implemented via async two-way webhook bridge)
- [ ] Persist/clean up generated images (`output/` dir, storage policy)
- [ ] Address `torch.load(weights_only=False)` FutureWarning on the SCHP checkpoint before PyTorch flips that default (add the checkpoint class to `torch.serialization.add_safe_globals`, or pin torch)
- [ ] Decide on GPU/hosting story for real usage — MPS locally is fine for dev/testing, but 384×512 is a reduced resolution and a Mac won't be a real production inference target

### Architecture decision still open
An earlier draft diagram proposed: Client Browser → Macbook (web server) → AES/RSA encrypt → transfer over SSH/WiFi → **Friend's Laptop** (runs the actual AI generation) → transfer back → Macbook → Client. Flagged issues at the time:
- Friend's laptop as the AI compute node is a single point of failure (availability, latency) — conflicts with just running CatVTON locally via MPS, which is what actually got built and works this session.
- The AES/RSA encryption layer looked redundant given SSH already encrypts in transit, unless there's a specific at-rest requirement.
- The diagram's return path had no re-encryption step, and some boxes ("Web Server hosting Website", "Web Shopper Interface") were disconnected/orphaned in the flow.
Recommendation from that discussion: prefer running generation locally over depending on a second machine, unless there's a real reason (e.g. the Mac genuinely can't handle the compute load).

---

## Context Dump

*(paste anything here — requirements, links, decisions, constraints, whatever. Doesn't need structure.)*

