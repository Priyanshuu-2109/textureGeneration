"""
Test script to verify all models can be instantiated and run forward passes.
"""
import torch
import yaml

from models import HierarchicalGenerator, GlobalDiscriminator, PatchDiscriminator
from models import HybridGenerator, HybridDiscriminator
from models import DiffusionModel


def test_hierarchical_gan():
    """Test Hierarchical GAN models."""
    print("Testing Hierarchical GAN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = HierarchicalGenerator(
        latent_dim=128,
        global_channels=64,
        local_channels=128,
        final_size=256
    ).to(device)
    
    global_disc = GlobalDiscriminator(base_channels=64).to(device)
    patch_disc = PatchDiscriminator(base_channels=64, patch_size=70).to(device)
    
    # Test generator
    global_noise = torch.randn(2, 128, device=device)
    local_noise = torch.randn(2, 3, 256, 256, device=device)
    output, global_output = generator(global_noise, local_noise)
    assert output.shape == (2, 3, 256, 256), f"Expected (2, 3, 256, 256), got {output.shape}"
    assert global_output.shape == (2, 3, 64, 64), f"Expected (2, 3, 64, 64), got {global_output.shape}"
    
    # Test discriminators
    fake_global = global_output
    fake_patch = output[:, :, :70, :70]
    
    global_pred = global_disc(fake_global)
    patch_pred = patch_disc(fake_patch)
    
    assert global_pred.shape == (2,), f"Expected (2,), got {global_pred.shape}"
    assert patch_pred.shape == (2, 1, 1, 1), f"Expected (2, 1, 1, 1), got {patch_pred.shape}"
    
    print("✓ Hierarchical GAN tests passed!")


def test_hybrid_model():
    """Test Hybrid CNN-Transformer models."""
    print("Testing Hybrid CNN-Transformer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = HybridGenerator(
        latent_dim=128,
        base_channels=64,
        transformer_dim=256,
        num_heads=8,
        num_layers=4,
        image_size=256
    ).to(device)
    
    discriminator = HybridDiscriminator(
        base_channels=64,
        use_transformer=False
    ).to(device)
    
    # Test generator
    noise = torch.randn(2, 128, 256, 256, device=device)
    output = generator(noise)
    assert output.shape == (2, 3, 256, 256), f"Expected (2, 3, 256, 256), got {output.shape}"
    
    # Test discriminator
    pred = discriminator(output)
    assert pred.shape == (2,), f"Expected (2,), got {pred.shape}"
    
    print("✓ Hybrid CNN-Transformer tests passed!")


def test_diffusion_model():
    """Test Diffusion model."""
    print("Testing Diffusion Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DiffusionModel(
        image_size=256,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        unet_channels=64
    ).to(device)
    
    model.register_schedule(device)
    
    # Test forward pass
    x_start = torch.randn(2, 3, 256, 256, device=device)
    t = torch.randint(0, 1000, (2,), device=device).long()
    
    predicted_noise, noise = model(x_start, t)
    assert predicted_noise.shape == x_start.shape, f"Shape mismatch: {predicted_noise.shape} vs {x_start.shape}"
    assert noise.shape == x_start.shape, f"Shape mismatch: {noise.shape} vs {x_start.shape}"
    
    # Test sampling (smaller for speed)
    print("  Testing sampling (this may take a moment)...")
    samples = model.p_sample_loop((1, 3, 64, 64), device)  # Smaller size for testing
    assert samples.shape == (1, 3, 64, 64), f"Expected (1, 3, 64, 64), got {samples.shape}"
    
    print("✓ Diffusion Model tests passed!")


def main():
    print("=" * 50)
    print("Testing Texture Synthesis Models")
    print("=" * 50)
    
    try:
        test_hierarchical_gan()
        test_hybrid_model()
        test_diffusion_model()
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
