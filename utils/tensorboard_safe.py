"""Safe TensorBoard import that falls back to no-op when TensorFlow is broken."""

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    # Fallback when TensorBoard/TensorFlow fails (e.g. broken TF on Mac ARM)
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_image(self, *args, **kwargs):
            pass

        def close(self):
            pass
