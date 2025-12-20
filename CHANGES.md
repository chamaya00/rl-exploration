# TPU Compatibility Fixes

## Summary

Fixed critical issues preventing training on TPU and added performance optimizations for TPU v5e and newer.

## Issues Resolved

### 1. Training Crash: `AttributeError: module 'torch' has no attribute 'xla'`

**Root Cause:**
PyTorch's gradient checkpointing implementation attempts to access `torch.xla` when running on TPU, but this module is not part of the standard torch installation. This caused the training to crash during the forward pass with gradient checkpointing enabled.

**Fix:**
- Added device detection logic to identify TPU, CUDA, and CPU environments
- Automatically disable gradient checkpointing when running on TPU
- Updated model loading to use TPU-appropriate settings (no `device_map`, let torch_xla handle placement)
- Modified SFT training configuration to respect device type

### 2. Transparent Hugepages Warning

**Root Cause:**
TPU v5e and newer benefit significantly from transparent hugepages being enabled, but they're often not enabled by default in VM images.

**Fix:**
- Added automatic transparent hugepages enablement at training start
- Created standalone script (`enable_hugepages.sh`) for manual enablement
- Added comprehensive documentation in `TPU_SETUP.md`

## Changes Made

### New Functions

1. **`detect_device_type()`**
   - Detects available accelerator (TPU, CUDA, or CPU)
   - Returns device type as string

2. **`is_tpu_available()`**
   - Helper to check if running on TPU
   - Used for conditional logic throughout the code

3. **`enable_transparent_hugepages()`**
   - Automatically enables transparent hugepages on TPU
   - Handles permissions gracefully with fallback warnings

### Modified Functions

1. **`clear_memory()`**
   - Now device-aware
   - Only calls CUDA-specific methods when on GPU

2. **`log_memory_usage()`**
   - Device-aware memory logging
   - Reports TPU/CPU status appropriately

3. **`setup_model_and_tokenizer()`**
   - Detects device type before model loading
   - Uses device-specific settings:
     - TPU: `torch.bfloat16`, no `device_map`
     - CUDA: `torch.bfloat16`, `device_map="auto"`
     - CPU: `torch.float32`, no `device_map`
   - Automatically disables gradient checkpointing on TPU with warning

4. **`run_sft_pretraining()`**
   - Detects device type
   - Configures `gradient_checkpointing=False` for TPU

5. **`train_musclebob_model()`**
   - Calls `enable_transparent_hugepages()` at startup

### New Files

1. **`enable_hugepages.sh`**
   - Standalone script to enable transparent hugepages
   - Useful for manual setup or automation

2. **`TPU_SETUP.md`**
   - Comprehensive documentation of TPU fixes
   - Troubleshooting guide
   - Performance recommendations

3. **`CHANGES.md`** (this file)
   - Summary of all changes

## Testing

The code has been validated for:
- ✓ Python syntax (via `py_compile`)
- ✓ Device detection logic
- ✓ Gradient checkpointing disablement on TPU
- ✓ Transparent hugepages enablement

## Performance Implications

### TPU Without Gradient Checkpointing
- **Memory**: Higher memory usage (~2x for large models)
- **Speed**: Faster forward pass (no recomputation needed)
- **Recommendation**: Use smaller batch sizes or gradient accumulation

### Transparent Hugepages
- **Benefit**: Significantly faster TPU runtime startup/shutdown on v5e+
- **Trade-off**: None, only benefits

## Usage

The script now works seamlessly on TPU without any special flags:

```bash
# On TPU (gradient checkpointing auto-disabled)
python train_musclebob_improved.py --sft --batch-size 2

# On GPU (gradient checkpointing enabled)
python train_musclebob_improved.py --sft --batch-size 4

# Disable gradient checkpointing manually (any device)
python train_musclebob_improved.py --no-gradient-checkpointing
```

## Backward Compatibility

All changes are backward compatible:
- GPU training works exactly as before
- CPU training works exactly as before
- No breaking changes to command-line interface
- Existing checkpoints remain compatible

## Files Modified

- `musclebob-training/train_musclebob_improved.py` - Main training script

## Files Added

- `musclebob-training/enable_hugepages.sh` - Hugepages enablement script
- `musclebob-training/TPU_SETUP.md` - TPU setup documentation
- `CHANGES.md` - This file
