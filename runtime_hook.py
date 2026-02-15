#!/usr/bin/env python3
"""
Runtime hook for PyInstaller to set environment variables before the application starts.
This resolves OpenMP conflicts when multiple libraries bring their own OpenMP implementations.
"""

import os
import sys
import multiprocessing

# Set KMP_DUPLICATE_LIB_OK to allow multiple OpenMP libraries
# This is needed because different packages (numpy, scipy, ctranslate2, onnxruntime)
# may bring different OpenMP implementations (libiomp5md.dll vs mk2iomp5md.dll)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Suppress transformers advisory warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Configure ONNX Runtime to use half of available CPU cores for better performance
# This prevents oversubscription and resource contention
cpu_count = multiprocessing.cpu_count()
optimal_threads = max(1, cpu_count // 2)

# Set ONNX Runtime environment variables for CPU execution
os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
os.environ['MKL_NUM_THREADS'] = str(optimal_threads)

print(f"Runtime hook: Set KMP_DUPLICATE_LIB_OK=TRUE to resolve OpenMP conflicts")
print(f"Runtime hook: Set TRANSFORMERS_NO_ADVISORY_WARNINGS=1 to suppress advisory warnings")
print(f"Runtime hook: Configured ONNX Runtime to use {optimal_threads} threads (half of {cpu_count} available CPUs)")