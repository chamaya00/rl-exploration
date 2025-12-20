# TPU Training Setup Guide

This document explains the TPU-specific fixes and optimizations applied to the training script.

## Issues Fixed

### 1. Gradient Checkpointing Incompatibility with TPU/XLA

**Problem:**
```
AttributeError: module 'torch' has no attribute 'xla'
```

PyTorch's gradient checkpointing implementation tries to access `torch.xla` when running on TPU, but the standard torch installation doesn't include this module. This causes training to crash.

**Solution:**
The training script now:
- Detects the device type (CUDA, TPU, or CPU)
- Automatically disables gradient checkpointing when running on TPU
- Logs a warning to inform users about this limitation

**Code Changes:**
- Added `detect_device_type()` and `is_tpu_available()` functions
- Modified `setup_model_and_tokenizer()` to disable gradient checkpointing on TPU
- Updated `run_sft_pretraining()` to use device-aware configuration

### 2. Transparent Hugepages Warning

**Problem:**
```
UserWarning: Transparent hugepages are not enabled. TPU runtime startup and
shutdown time should be significantly improved on TPU v5e and newer.
```

**Solution:**
The training script now automatically attempts to enable transparent hugepages when running on TPU. This improves TPU runtime performance.

**Manual Enable (if automatic fails):**
```bash
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
```

Or use the provided script:
```bash
./enable_hugepages.sh
```

### 3. Device-Specific Model Loading

**Changes:**
- **CUDA/GPU**: Uses `device_map="auto"` and `torch.bfloat16`
- **TPU**: Uses `torch.bfloat16` without device_map (torch_xla handles placement)
- **CPU**: Uses `torch.float32` for compatibility

## Device Detection

The script automatically detects the available accelerator:

```python
def detect_device_type():
    # Check for TPU
    try:
        import torch_xla.core.xla_model as xm
        return 'tpu'
    except ImportError:
        pass

    # Check for CUDA
    if torch.cuda.is_available():
        return 'cuda'

    return 'cpu'
```

## Training on TPU

The script is now fully compatible with TPU training:

1. **No gradient checkpointing** - Automatically disabled to prevent XLA errors
2. **Transparent hugepages** - Automatically enabled for better performance
3. **Proper device placement** - torch_xla handles device placement automatically
4. **Memory management** - TPU-aware memory logging and cleanup

## Performance Notes

### Memory Usage
- Without gradient checkpointing on TPU, memory usage will be higher
- Consider reducing batch size if you encounter OOM errors
- Use gradient accumulation to compensate for smaller batch sizes

### Recommended Settings for TPU
```bash
python train_musclebob_improved.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 2 \
  --epochs 5 \
  --num-generations 8 \
  --sft \
  --sft-epochs 2
```

## Troubleshooting

### Still seeing torch.xla errors?
Make sure you're using the latest version of the script and that the device detection is working:
```python
from train_musclebob_improved import detect_device_type
print(f"Detected device: {detect_device_type()}")
```

### Hugepages not enabling?
Try manually with sudo:
```bash
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
```

### Out of memory on TPU?
Reduce batch size and increase gradient accumulation:
```bash
python train_musclebob_improved.py --batch-size 1 --epochs 5
```

## References

- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [TPU Best Practices](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm)
- [Transparent Hugepages](https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html)
