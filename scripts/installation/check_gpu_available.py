# check pytorch, tensorflow and jax
import torch
import tensorflow as tf
import jax


def check_gpu_available():
    gpu_available = torch.cuda.is_available()
    assert gpu_available, "GPU is not available for PyTorch."

    try:
        tf.config.list_physical_devices('GPU')
    except RuntimeError as e:
        raise RuntimeError("TensorFlow GPU check failed: " + str(e))

    jax_gpu_available = jax.devices('gpu')
    assert len(jax_gpu_available) > 0, "GPU is not available for JAX."

    print("GPU is available for PyTorch, TensorFlow, and JAX.")


if __name__ == "__main__":
    check_gpu_available()