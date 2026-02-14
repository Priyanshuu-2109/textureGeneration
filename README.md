# Novel Multi-Scale Architectures for Texture Synthesis

This repository implements advanced texture synthesis architectures designed to generate high-quality, non-repetitive textures from limited training data (~40 images).

## Features

- **Hierarchical Global-Local GANs**: Two-stage generator with global structure modeling and local detail refinement
- **Convolution + Transformer Hybrid**: Self-attention layers for long-range dependencies with convolutional detail synthesis
- **Diffusion-Based Texture Model**: Denoising diffusion model for diverse, artifact-free texture generation
- **Advanced Training Strategies**: Progressive resizing, heavy augmentation, multi-scale losses
- **Comprehensive Evaluation**: SIFID, LPIPS, seam continuity metrics

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Prepare your texture dataset**: Place your ~40 texture images in a `textures/` directory. Supported formats: JPG, PNG, BMP.

2. **Configure training**: Edit `config.yaml` to adjust hyperparameters, model settings, and paths.

3. **Train a model**:

```bash
# Hierarchical Global-Local GAN (Recommended for most textures)
python train_hierarchical_gan.py --data_dir ./textures --config config.yaml

# Convolution + Transformer Hybrid (Good for textures with global patterns)
python train_hybrid_model.py --data_dir ./textures --config config.yaml

# Diffusion Model (Best diversity, slower training)
python train_diffusion.py --data_dir ./textures --config config.yaml
```

4. **Generate samples**:

```bash
python generate.py \
    --model_path ./checkpoints/checkpoint_epoch_200.pth \
    --model_type hierarchical \
    --output_dir ./outputs \
    --num_samples 10 \
    --size 256
```

5. **Evaluate results**:

```bash
python evaluate.py \
    --model_path ./checkpoints/checkpoint_epoch_200.pth \
    --model_type hierarchical \
    --data_dir ./textures \
    --num_samples 50
```

## Architecture Details

### 1. Hierarchical Global-Local GANs

Splits generation into two stages:
- **Global Module**: Generates coarse structure at low resolution (64×64)
- **Local Refinement Module**: Adds fine texture details at high resolution (256×256)
- Uses two discriminators: Global Discriminator for large-scale realism, PatchGAN for local texture fidelity

### 2. Convolution + Transformer Hybrid

Combines:
- **CNN layers**: Extract local patterns and details
- **Transformer blocks**: Capture long-range texture correlations via self-attention
- Enables the model to learn both "what" (texture elements) and "where" (spatial relationships)

### 3. Diffusion-Based Texture Model

- Denoising diffusion process trained on patches from multiple images
- U-Net backbone with large receptive field for patch-level statistics
- Naturally models multi-scale statistics through iterative denoising
- Produces highly diverse, artifact-free samples

## Training Tips

- **Progressive Resizing**: Enabled by default - starts at 64×64 and gradually increases to final resolution
- **Heavy Augmentation**: Random crops, flips, color jitter, and rotation multiply effective dataset size
- **Auxiliary Losses**: Style loss (Gram matrices), FFT loss (power spectrum), and perceptual loss improve quality
- **Limited Data**: With ~40 images, use aggressive augmentation and consider pretraining on larger texture datasets

## Configuration

Key parameters in `config.yaml`:

- `data.image_size`: Final output resolution (256 recommended)
- `training.batch_size`: Adjust based on GPU memory (8-16 typical)
- `losses.style_weight`: Weight for style loss (10.0 default)
- `training.progressive_resizing`: Enable/disable progressive training

## Evaluation Metrics

- **SIFID**: Single-Image FID - measures perceptual quality and diversity
- **LPIPS**: Learned Perceptual Image Patch Similarity - perceptual distance
- **Seam Continuity**: Measures border artifacts in tiled outputs (lower is better)

## Citation

If you use this code, please cite the research document describing these architectures.

## License

MIT License - see LICENSE file for details.
