#!/bin/bash

# GPU Setup Script
# Installs ONNX Runtime GPU for faster performance

echo "=========================================="
echo "GPU Setup for Attendance System"
echo "=========================================="
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "❌ No NVIDIA GPU detected"
    echo "   GPU acceleration requires an NVIDIA GPU with CUDA support"
    echo ""
    read -p "Continue anyway? (y/N): " continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA installation
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA Toolkit installed:"
    nvcc --version | grep "release"
    echo ""
else
    echo "⚠️  CUDA Toolkit not found"
    echo "   Download from: https://developer.nvidia.com/cuda-downloads"
    echo ""
fi

echo "Installing ONNX Runtime GPU..."
echo ""

# Uninstall CPU version
pip uninstall -y onnxruntime

# Install GPU version
pip install onnxruntime-gpu

echo ""
echo "=========================================="
echo "Verifying GPU installation..."
echo "=========================================="
echo ""

# Run GPU check
python3 check_gpu.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Run the attendance system with:"
echo "  ./run_rtsp_camera.sh"
echo ""
