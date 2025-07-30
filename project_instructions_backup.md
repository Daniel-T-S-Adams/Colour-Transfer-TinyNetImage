# Goal
Use two Tiny ImageNet images to build empirical RGB distributions (one point per pixel, equally weighted), compute the optimal transport (OT) plan between them using a **stabilized log-domain Sinkhorn algorithm**, and apply the transport plan to perform **colour transfer** from the source image to the target image.

These instructions are written so you can hand them to an AI coding assistant. They are explicit about inputs/outputs, steps, and checks.

---
## High-level Steps
1. **Download & unpack Tiny ImageNet.**
2. **Select two images** (source & target) and load them as 64×64 RGB arrays.
3. **Construct empirical distributions** in \(\mathbb{R}^3\) from pixel colours.
4. **Compute the OT plan** with an entropic-regularized, numerically stabilized log-Sinkhorn algorithm.
5. **Perform colour transfer** using the OT plan (barycentric projection or stochastic map) and reconstruct the recoloured image.
6. **Validate & visualize** results.
7. **Package outputs** (code, figures, logs) for reproducibility.

---
## Detailed Instructions

### 0. Environment & Libraries ✅ COMPLETED
- Language: Python 3.10+ (unless you prefer another; adapt accordingly).
- Required libs: `numpy`, `scipy`, `Pillow` or `imageio`, `matplotlib`, optionally `pot` (Python Optimal Transport) for cross-checking but **implement Sinkhorn yourself**.
- Set a fixed random seed for reproducibility.

### 1. Download Tiny ImageNet (64×64) ✅ COMPLETED
**Task**: Write code that:
- Downloads Tiny ImageNet from a reliable mirror (e.g., the original Stanford CS231n link) using `wget`/`requests`. ✅ DONE
- Unzips it and exposes the folder structure (train/val/test with synsets). ✅ DONE
- Confirms image size = 64×64 and 3 channels, converting if needed. ✅ DONE

**Outputs**:
- Local path to dataset root. ✅ DONE
- Function `load_tiny_imagenet_image(path: str) -> np.ndarray` returning an array of shape `(64,64,3)` with `dtype=float32` scaled to `[0,1]`. ✅ DONE

### 2. Pick Two Images ✅ COMPLETED
**Task**: Implement a helper to randomly or deterministically pick two images. This is done in the image_selector.py
- Let `img_src` be the **source** (whose colours you want to transfer to target or vice versa — decide and state clearly). 
- Let `img_tgt` be the **target**. 

**Outputs**:
- `img_src`, `img_tgt` arrays and their file paths. ✅ DONE
- Visualize both with matplotlib and save to disk for the report. ✅ DONE

### 3. Build Empirical RGB Distributions ✅ COMPLETED
Treat each pixel colour (R,G,B) as a point in \(\mathbb{R}^3\).

**Task**:
- Reshape each image to `(N, 3)` where `N = 64*64 = 4096`. ✅ DONE
- Normalize RGB to `[0,1]` if not already. ✅ DONE
- Define uniform weights: `w = np.ones(N)/N` for both images. ✅ DONE

**Outputs**:
- `X_src ∈ ℝ^{N×3}`, `w_src ∈ ℝ^{N}` ✅ DONE
- `X_tgt ∈ ℝ^{N×3}`, `w_tgt ∈ ℝ^{N}` ✅ DONE

### 4. Compute Cost Matrix ❌ NOT COMPLETED
Use squared Euclidean distance in RGB space. Since its a 

**Task**:
- `C = ||X_src[i]-X_tgt[j]||^2` producing an `(N×N)` matrix. For `N=4096`, memory is ~16M entries (~128 MB float64). If needed, chunk/batch to reduce memory. Take advantage of the fact that its a symmetric matrix to save memory.
- Store in `float32` if safe.

### 5. Stabilized Log-domain Sinkhorn Algorithm ❌ NOT COMPLETED
Implement entropic OT with regularization parameter `ε` (epsilon). Use log-stabilization to prevent numerical under/overflow.

**Mathematical Formulation**:
Minimize \(⟨P,C⟩ + ε \sum_{ij} P_{ij}(\log P_{ij} - 1)\) s.t. `P 1 = w_src`, `Pᵀ 1 = w_tgt`, `P ≥ 0`.

**Algorithm (log-domain, with periodic stabilization)**:
1. Precompute `K = -C/ε` (do **not** exponentiate globally).
2. Initialize dual variables `f = 0`, `g = 0` (log scaling vectors for rows/cols).
3. Iterate until convergence:
   - Update `f`:
     \[
     f ← ε ( \log w_src - \log\sum_j \exp((K + g[None,:] + f[:,None])/ε)_j ) + f
     \]
   - Update `g` analogously for columns.
   - Every `t_stab` iterations, **stabilize** by absorbing the max into `f`/`g` to keep numbers centered (common trick: subtract mean or max).
   - Stopping criteria: max violation of marginals < `tol` (e.g., 1e-6) or max iterations reached.

**Practical Implementation Pattern**:
- Work with `log_u`, `log_v` vectors (or `f`, `g`).
- Use `logsumexp` for numerical stability (`scipy.special.logsumexp`).
- Keep everything in log-space; only exponentiate for the final plan or on-demand.
- Optional: implement **debiased / epsilon-scaling** (start high ε, gradually reduce) for sharper plans.

**Outputs**:
- Transport plan `P` (dense or sparse); optionally keep as implicit via `f`,`g`.
- Diagnostics: iterations, final marginal error, runtime.

### 6. Colour Transfer Using the Transport Plan ❌ NOT COMPLETED
Two common methods:

**(A) Barycentric Mapping (deterministic)**
For each source pixel i:
\[
\hat{y}_i = \frac{\sum_j P_{ij} X_{tgt}[j]}{\sum_j P_{ij}}
\]
Replace pixel i in the source image with \(\hat{y}_i\).

**(B) Stochastic Sampling**
For each i, sample j from categorical distribution proportional to `P_{ij}` and assign `X_tgt[j]`.

**Task**:
- Implement both; compare smoothness vs. noise.
- Clip results to `[0,1]`, reshape back to `(64,64,3)`, convert to `uint8` for saving.

**Outputs**:
- Recoloured image(s) saved to disk.
- Side-by-side plot: original source, target, transferred.

### 7. Validation & Checks ❌ NOT COMPLETED
- Verify `P` marginals: `P @ 1 ≈ w_src` and `Pᵀ @ 1 ≈ w_tgt`.
- Report `||P1 - w_src||₁` and `||Pᵀ1 - w_tgt||₁`.
- Sensitivity to ε: show effect on colour transfer.
- Optional: compare with POT’s `ot.sinkhorn` for sanity.

### 8. Packaging & Reproducibility ❌ NOT COMPLETED
- Wrap code in functions or a notebook with clear sections.
- Save: parameters (ε, tol, iters), final images, and plots.
- Provide a CLI or notebook cells to re-run with different images.

---
## Deliverables (what the AI should give back)
1. **Python code** implementing all steps (dataset download → colour transfer), well-commented.
2. **Documentation** within the code (docstrings) and a short README-like explanation.
3. **Figures**: original images and recoloured outputs.
4. **Logs/metrics** of Sinkhorn convergence.

---
## Prompt Template (copy/paste to your AI)
> You are an expert in Optimal Transport and scientific Python. Implement the following project end-to-end:
> 1. Download and unpack Tiny ImageNet (64×64 images).
> 2. Load two images (source & target) as 64×64×3 arrays scaled to [0,1].
> 3. Build empirical RGB distributions: reshape to (4096,3) and assign uniform weights.
> 4. Compute the squared Euclidean cost matrix between all pixel pairs.
> 5. Implement from scratch a stabilized log-domain entropic Sinkhorn algorithm (with epsilon, tol, max_iter, periodic stabilization) to solve OT between the two distributions. Provide convergence diagnostics.
> 6. Recover the transport plan and perform colour transfer via (a) barycentric mapping and (b) stochastic sampling. Output recoloured images.
> 7. Validate marginals and report errors, iteration counts, and runtime. Optionally compare with POT for sanity.
> 8. Produce plots (before/after images, maybe histograms of RGB distributions) and save all artifacts.
> 9. Package everything in a single notebook or script with clear functions and docstrings.
>
> Constraints:
> - Use only numpy/scipy/matplotlib/Pillow (and optionally POT for comparison).
> - Keep numeric operations stable (logsumexp, regular re-centering of dual variables).
> - Code must be reproducible (set seeds) and modular.
>
> Deliver: the full code, explanation of each step, and saved output images.

---
## Optional Enhancements
- **Epsilon annealing**: start ε large (e.g., 0.1) then reduce to 0.01, 0.005.
- **GPU acceleration** with JAX or PyTorch.
- **Use CIELab colour space** for perceptual uniformity instead of RGB.
- **Multi-image transfer** by averaging transport plans.
- **Memory optimizations**: blockwise Sinkhorn or low-rank approximations (Nyström).

---
## Acceptance Criteria Checklist
- [x] Dataset successfully downloaded and verified.
- [x] Two images loaded, visualized.
- [x] Distributions built correctly (shape and normalization).
- [ ] Sinkhorn implemented in log-space, converges within tolerance.
- [ ] Transport plan respects marginals.
- [ ] Colour transfer outputs look plausible and saved.
- [x] Code/documentation is clean and reproducible. *(partial - basic structure and docstrings present)*

---
Feel free to trim or expand sections depending on the AI’s capabilities.
