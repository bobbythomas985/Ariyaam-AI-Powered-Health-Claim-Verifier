# test_imports.py
import sys

required = [
    "torch", "transformers", "datasets", "peft", "accelerate", "trl",
    "bitsandbytes", "sentence_transformers", "pandas", "numpy", "scipy",
    "matplotlib", "seaborn", "tqdm", "huggingface_hub", "requests"
]

missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n⚠️ Missing packages: {missing}")
    sys.exit(1)
else:
    print("\n🎉 All required packages installed successfully!")
    print(f"CUDA available: {__import__('torch').cuda.is_available()}")
    if __import__('torch').cuda.is_available():
        vram = __import__('torch').cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU VRAM: {vram:.1f}GB")
