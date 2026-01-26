#!/usr/bin/env python3
"""
GPU Check Script
Verifies if GPU is available for ONNX Runtime
"""

import sys

def check_gpu():
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)
    print()
    
    # Check ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime version: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"\n📋 Available providers: {providers}")
        
        if "CUDAExecutionProvider" in providers:
            print("\n✅ GPU (CUDA) is AVAILABLE!")
            print("   Your models will run on GPU")
            
            # Get CUDA device info
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"\n🎮 GPU Device: {torch.cuda.get_device_name(0)}")
                    print(f"   CUDA Version: {torch.version.cuda}")
                    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            except ImportError:
                print("\n   (Install PyTorch to see detailed GPU info)")
                
        else:
            print("\n❌ GPU (CUDA) is NOT available")
            print("   Models will run on CPU")
            print("\n💡 To enable GPU:")
            print("   1. Install CUDA Toolkit")
            print("   2. Install: pip install onnxruntime-gpu")
            print("   3. Uninstall CPU version: pip uninstall onnxruntime")
            
    except ImportError:
        print("❌ ONNX Runtime not installed")
        print("   Install: pip install onnxruntime-gpu")
        sys.exit(1)
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    check_gpu()
