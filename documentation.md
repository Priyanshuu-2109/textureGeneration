# Texture Synthesis Architectures — Detailed Technical Documentation

This document describes the three texture synthesis architectures: tensor dimensions at each step, forward and backward passes, and the intuition behind each component.

---

## Table of Contents

1. [Hierarchical Global–Local GAN](#1-hierarchical-globallocal-gan)
2. [Convolution + Transformer Hybrid](#2-convolution--transformer-hybrid)
3. [Diffusion-Based Texture Model](#3-diffusion-based-texture-model)

---

# 1. Hierarchical Global–Local GAN

## 1.1 Overview

The generator is split into:

- **Global Module**: Produces a low-resolution structural map (large-scale flow, veins, layout).
- **Local Refinement Module**: Takes the upsampled global map plus local noise and adds high-resolution texture.

Two discriminators:

- **Global Discriminator**: Operates on low-res (e.g. 64×64) to enforce large-scale realism.
- **Patch Discriminator**: Operates on random high-res patches (e.g. 70×70) to enforce local texture.

## 1.2 System Inputs

| Input | Shape | Description |
|-------|--------|-------------|
| `global_noise` | `(B, latent_dim)` | Latent vector, e.g. `(B, 128)`. Drives global structure. |
| `local_noise` | `(B, 3, H, W)` | Spatial noise with same resolution as target, e.g. `(B, 3, 256, 256)`. Drives local detail. |

If `local_noise` is omitted, it is sampled as `randn_like(global_upsampled)` so its spatial size matches the upsampled global map (e.g. 256×256).

## 1.3 Global Module — Forward Pass

**Role:** Map a latent vector to a low-resolution RGB structure map.

| Step | Operation | Input shape | Output shape | Notes |
|------|-----------|-------------|-------------|--------|
| 1 | `latent_proj` (Linear + ReLU) | `(B, 128)` | `(B, 4096)` | 4096 = 64×4×4×4. Prepares latent for spatial decoding. |
| 2 | `view(b, -1, 4, 4)` | `(B, 4096)` | `(B, 256, 4, 4)` | Interpret as 256-channel 4×4 feature map. |
| 3 | `init_conv` (ConvTranspose2d 4×4, s=1, p=0) | `(B, 256, 4, 4)` | `(B, 256, 7, 7)` | Out = (4−1)×1 + 4 = 7. |
| 4 | `dec1` (ConvTranspose2d 4×4, s=2, p=1) | `(B, 256, 7, 7)` | `(B, 128, 14, 14)` | Double resolution. |
| 5 | `dec2` (ConvTranspose2d 4×4, s=2, p=1) | `(B, 128, 14, 14)` | `(B, 64, 28, 28)` | |
| 6 | `dec3` (ConvTranspose2d 4×4, s=2, p=1) + Tanh | `(B, 64, 28, 28)` | `(B, 3, 56, 56)` | **Global structure output.** |

**Intuition:** The latent vector is expanded into a small 4×4 grid, then transposed convolutions double resolution repeatedly. The network learns to decode global layout (e.g. vein direction, coarse pattern) at 56×56. No spatial conditioning beyond the latent.

## 1.4 Hierarchical Generator — Full Forward Pass

| Step | Operation | Tensor shape | Notes |
|------|-----------|--------------|--------|
| 1 | `global_module(global_noise)` | `(B, 128)` → `(B, 3, 56, 56)` | Low-res structure. |
| 2 | `F.interpolate(..., size=(final_size, final_size))` | `(B, 3, 56, 56)` → `(B, 3, 256, 256)` | Bilinear upsampling to target resolution. |
| 3 | `local_noise` (given or sampled) | — | `(B, 3, 256, 256)`. Must match spatial size of upsampled global. |
| 4 | `local_module(global_upsampled, local_noise)` | See below | Refinement stage. |

## 1.5 Local Refinement Module — Forward Pass

**Role:** Combine upsampled global structure and local noise into a high-res texture, with residual connection to the global map.

| Step | Operation | Input shape | Output shape |
|------|-----------|-------------|-------------|
| 1 | `torch.cat([global_upsampled, local_noise], dim=1)` | Two `(B, 3, 256, 256)` | `(B, 6, 256, 256)` |
| 2 | `conv1` (3×3, 6→128) + BN + ReLU | `(B, 6, 256, 256)` | `(B, 128, 256, 256)` |
| 3 | `conv2` (3×3) + residual `+ x` | `(B, 128, 256, 256)` | `(B, 128, 256, 256)` |
| 4 | `conv3` (3×3) + residual `+ x` | `(B, 128, 256, 256)` | `(B, 128, 256, 256)` |
| 5 | `conv4` (3×3) | `(B, 128, 256, 256)` | `(B, 128, 256, 256)` |
| 6 | `output` (3×3, 128→3) + Tanh | `(B, 128, 256, 256)` | `(B, 3, 256, 256)` |
| 7 | `+ global_upsampled` | `(B, 3, 256, 256)` + `(B, 3, 256, 256)` | `(B, 3, 256, 256)` **Final texture** |

**Intuition:** The 6-channel input carries both “what” (global layout) and “where to add variation” (local noise). Convolutions and residuals fill in fine detail while the residual add keeps the global structure stable.

## 1.6 Global Discriminator — Forward Pass

**Input:** Low-res image, e.g. `(B, 3, 64, 64)` (training often uses 64×64 crops or downsampled images).

| Layer | Op | Shape in | Shape out |
|-------|-----|----------|-----------|
| Conv2d 4×4, s=2, p=1 | 3 → 64 | (B,3,64,64) | (B,64,32,32) |
| Conv2d 4×4, s=2, p=1 | 64 → 128 | (B,64,32,32) | (B,128,16,16) |
| Conv2d 4×4, s=2, p=1 | 128 → 256 | (B,128,16,16) | (B,256,8,8) |
| Conv2d 4×4, s=2, p=1 | 256 → 512 | (B,256,8,8) | (B,512,4,4) |
| Conv2d 4×4, s=1, p=0 | 512 → 1 | (B,512,4,4) | (B,1,1,1) |
| `view(B,-1).mean(1)` | — | (B,1,1,1) | **(B,)** |

**Intuition:** Standard CNN classifier over the whole low-res image; one scalar per batch item. Encourages globally coherent structure.

## 1.7 Patch Discriminator — Forward Pass

**Input:** Patch of size at least 70×70 (e.g. `(B, 3, 70, 70)` or a random 70×70 crop from a larger image).

| Layer | Shape in | Shape out |
|-------|----------|-----------|
| Same 5 conv blocks as Global Disc | (B,3,70,70) | (B,1,4,4) |

Output is a small spatial map `(B, 1, 4, 4)` (or similar); training code may aggregate (e.g. mean over spatial). **Intuition:** PatchGAN judges local patches, so the generator must match local texture statistics.

## 1.8 Backward Pass (Training)

- **Generator:** Receives gradients from (1) Global Discriminator on `global_output`, (2) Patch Discriminator on `fake_output`, (3) optional auxiliary losses (e.g. style, FFT, perceptual). Gradients flow: discriminators → `fake_output` / `global_output` → Local Refinement Module and Global Module → `global_noise` and `local_noise`.
- **Global Discriminator:** Trained to classify real vs fake low-res images; gradients only update discriminator parameters.
- **Patch Discriminator:** Trained to classify real vs fake patches; gradients only update patch discriminator parameters.

Auxiliary losses (e.g. VGG style, FFT) add more gradient paths into the generator; they do not change the tensor shapes above.

---

# 2. Convolution + Transformer Hybrid

## 2.1 Overview

- **CNN encoder:** Builds local feature maps from spatial noise.
- **Patch embedding:** Flattens spatial grid into a sequence of patches and projects to transformer dimension.
- **Transformer blocks:** Self-attention across patches for global texture correlation.
- **Patch unembed + CNN decoder:** Sequence back to spatial feature map, then convolutions to RGB.

## 2.2 System Input

| Input | Shape | Description |
|-------|--------|-------------|
| `noise` | `(B, latent_dim, H, W)` | Spatial noise, e.g. `(B, 128, 256, 256)`. Treated as a 128-channel image. |

## 2.3 CNN Encoder — Forward Pass

Three `CNNBlock`s (Conv2d 3×3, stride 1, padding 1); no resolution change.

| Step | Block | Input shape | Output shape |
|------|--------|-------------|-------------|
| 1 | CNNBlock(128 → 64) | (B, 128, 256, 256) | (B, 64, 256, 256) |
| 2 | CNNBlock(64 → 128) | (B, 64, 256, 256) | (B, 128, 256, 256) |
| 3 | CNNBlock(128 → 256) | (B, 128, 256, 256) | (B, 256, 256, 256) |

**Intuition:** Convolutions extract local texture features at full resolution before global reasoning.

## 2.4 Patch Embedding — Forward Pass

- **Rearrange:** `(B, 256, 256, 256)` → treat as 16×16 grid of 16×16 patches, each with 256 channels.  
  Shape: `(B, num_patches, 16×16×256)` = `(B, 256, 65536)` (num_patches = (256/16)² = 256).
- **Linear(65536, 256):** `(B, 256, 65536)` → `(B, 256, 256)`.
- **Add positional embedding:** `(1, 256, 256)` broadcast to `(B, 256, 256)`.

**Intuition:** Each “token” is one 16×16 patch’s flattened features; position encoding tells the transformer where each patch lies.

## 2.5 Transformer Block — Forward Pass

**MultiHeadSelfAttention:**

| Step | Operation | Shape |
|------|-----------|--------|
| `to_qkv(x)` | Linear(dim → 3×inner_dim) | (B, 256, 256) → (B, 256, 1536) |
| chunk(3) → q,k,v | — | Each (B, 256, 512); heads=8, dim_head=64 |
| rearrange to (B, heads, n, d) | — | (B, 8, 256, 64) |
| attention: softmax(qk^T/√d)v | — | (B, 8, 256, 64) |
| rearrange + to_out | — | (B, 256, 256) |

**TransformerBlock:** `x + attn(norm1(x))` then `x + mlp(norm2(x))`. Sequence length and dimension stay `(B, 256, 256)`.

**Intuition:** Self-attention lets each patch depend on every other patch, capturing long-range structure (e.g. repeating stripes, global flow).

## 2.6 Patch Unembed + CNN Decoder — Forward Pass

| Step | Operation | Input shape | Output shape |
|------|-----------|-------------|-------------|
| 1 | Linear(256 → 16×16×256) | (B, 256, 256) | (B, 256, 65536) |
| 2 | Rearrange to spatial | (B, 256, 65536) | (B, 256, 256, 256) |
| 3 | CNNBlock(256 → 128) | (B, 256, 256, 256) | (B, 128, 256, 256) |
| 4 | CNNBlock(128 → 64) | (B, 128, 256, 256) | (B, 64, 256, 256) |
| 5 | Conv2d(64 → 3) + Tanh | (B, 64, 256, 256) | **(B, 3, 256, 256)** |

**Intuition:** Sequence is reshaped back to the spatial grid; convolutions refine and project to RGB.

## 2.7 Hybrid Discriminator — Forward Pass (no transformer)

| Stage | Operation | Shape in | Shape out |
|-------|------------|----------|-----------|
| cnn_features | 3× Conv2d 4×4 s=2 | (B,3,256,256) | (B,256,32,32) |
| final_conv | 2× Conv2d | (B,256,32,32) | (B,1,4,4) |
| view + mean(1) | — | (B,1,4,4) | **(B,)** |

**Intuition:** Standard CNN discriminator; optional transformer branch can be added for global discrimination.

## 2.8 Backward Pass (Hybrid)

Gradients flow: discriminator loss and any auxiliary losses → generator output → CNN decoder → patch unembed → transformer blocks → patch embed → CNN encoder → input noise. Autograd handles the full path; transformer and CNN parameters are updated together.

---

# 3. Diffusion-Based Texture Model

## 3.1 Overview

- **Forward (training):** Sample timestep `t`, add noise to real image to get `x_t`, U-Net predicts the noise; loss is MSE(predicted_noise, true_noise).
- **Sampling:** Start from pure noise `x_T`, then repeatedly run the U-Net to predict noise and step to `x_{t-1}` until `x_0`.

## 3.2 Inputs and Outputs

**Training:**

| Symbol | Shape | Description |
|--------|--------|-------------|
| `x_start` | (B, 3, H, W) | Real texture image (e.g. 256×256). |
| `t` | (B,) | Timestep indices in [0, T−1]. |
| `noise` | (B, 3, H, W) | Standard Gaussian. |
| `x_noisy` | (B, 3, H, W) | Noisy image at time t. |
| Predicted noise | (B, 3, H, W) | U-Net output. |

**Sampling:**

| Symbol | Shape | Description |
|--------|--------|-------------|
| `shape` | e.g. (B, 3, 256, 256) | Desired image shape. |
| `img` (initial) | (B, 3, 256, 256) | Pure Gaussian. |
| `img` (after loop) | (B, 3, 256, 256) | Generated texture. |

## 3.3 Forward Diffusion q(x_t | x_0)

\[
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon,\quad \varepsilon\sim\mathcal{N}(0,I).
\]

- `sqrt_alphas_cumprod[t]`, `sqrt_one_minus_alphas_cumprod[t]` are indexed by `t` (shape `(B,)`) and broadcast to `(B,1,1,1)`.
- **Tensor:** `x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise` → shape `(B, 3, H, W)`.

**Intuition:** Single step from clean image to noisy image at time `t`; no learned parameters.

## 3.4 U-Net — Forward Pass (Denoising Network)

**Time embedding:** `timestep` `(B,)` → SinusoidalPositionEmbeddings → Linear → SiLU → `(B, 128)`.

**Encoder (example for 256×256 input):**

| Block | Input | After block | After pool |
|-------|--------|-------------|------------|
| down1 | (B,3,256,256) | (B,64,256,256) | — |
| down2 | (B,64,128,128) | (B,128,128,128) | — |
| down3 | (B,128,64,64) | (B,256,64,64) | — |
| bottleneck | (B,256,32,32) | (B,512,32,32) | — |

Each block is a ResidualBlock with time embedding; pooling is 2×2 (halves resolution).

**Decoder (skip connections from encoder):**

| Block | Upsampled + skip | After block |
|-------|-------------------|-------------|
| up1 | concat(x4_up, x3) → (B,768,32,32) | (B,256,32,32) |
| up2 | concat(up1_up, x2) → (B,384,64,64) | (B,128,64,64) |
| up3 | concat(up2_up, x1) → (B,192,128,128) | (B,64,128,128) |
| output | interpolate to 256×256, then GroupNorm+SiLU+Conv2d | **(B, 3, 256, 256)** |

**Intuition:** Encoder captures multi-scale structure; skip connections preserve detail; time embedding lets the network behave differently per noise level.

## 3.5 Reverse Step p(x_{t-1} | x_t)

\[
\mu = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\varepsilon}\right),\quad
x_{t-1} = \mu + \sigma_t z,\ z\sim\mathcal{N}(0,I).
\]

- `model(x, t)` returns predicted noise `(B, 3, H, W)`.
- `model_mean` computed as in the formula; then add `sqrt(posterior_variance_t) * noise` when t>0.
- **Tensor:** `x` goes from `(B, 3, H, W)` to updated `(B, 3, H, W)`.

## 3.6 Sampling Loop

1. `img = randn(shape)` → (B, 3, 256, 256).
2. For t = T−1 down to 0: `img = p_sample(img, t)`.
3. Return `img` as generated image.

## 3.7 Backward Pass (Diffusion)

- **Training:** Loss = MSE(predicted_noise, noise). Gradients flow: loss → U-Net output → all U-Net layers and time embedding. No gradients through the fixed schedule (betas, alphas).
- **Sampling:** No backward; only forward passes in the loop.

---

# Summary Table: Output Shapes

| Architecture | Main generator input | Generator output | Discriminator input (typical) |
|--------------|---------------------|------------------|-------------------------------|
| Hierarchical GAN | global: (B,128), local: (B,3,256,256) | (B,3,256,256) | Global: (B,3,64,64); Patch: (B,3,70,70) |
| Hybrid | (B,128,256,256) | (B,3,256,256) | (B,3,256,256) |
| Diffusion | x_start (B,3,H,W), t (B,) | predicted_noise (B,3,H,W) | — (no discriminator) |

---

# Design Intuition Summary

- **Hierarchical GAN:** Separate “global layout” (low-res) and “local detail” (high-res) with two discriminators to avoid tile-like repetition and enforce both structure and texture.
- **Hybrid:** CNN for local texture, transformer for long-range correlation; one generator produces full-resolution output.
- **Diffusion:** No discriminator; learning is matching the noise prediction at every timestep; sampling is iterative denoising; good for diversity and stability with limited data.

This documentation reflects the current code (e.g. GlobalModule output 56×56, then upsampled to final_size). Any change in layer sizes or strides will change the dimensions accordingly.
