#!/usr/bin/env python3
"""
Test script to verify LoRA implementation
"""

import sys
import os


def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing LoRA dependencies...")

    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import transformers
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False

    try:
        import peft
        print("✅ PEFT imported successfully")
    except ImportError as e:
        print(f"❌ PEFT import failed: {e}")
        return False

    try:
        import bitsandbytes
        print("✅ BitsAndBytes imported successfully")
    except ImportError as e:
        print(f"❌ BitsAndBytes import failed: {e}")
        return False

    return True


def test_config():
    """Test if config file is valid"""
    print("\n🔧 Testing configuration...")

    try:
        import yaml
        with open("config.yaml", 'r') as file:
            config = yaml.safe_load(file)

        # Check if LoRA config exists
        if 'lora' in config:
            print("✅ LoRA configuration found")
            lora_config = config['lora']
            required_keys = ['enabled', 'r', 'lora_alpha', 'target_modules']
            for key in required_keys:
                if key in lora_config:
                    print(f"✅ {key}: {lora_config[key]}")
                else:
                    print(f"❌ Missing {key} in LoRA config")
                    return False
        else:
            print("❌ No LoRA configuration found")
            return False

        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_model_loading():
    """Test if we can load a model with LoRA"""
    print("\n🤖 Testing model loading with LoRA...")

    try:
        from fine_tuning_lora import RobertoLoRAFineTuner

        # Initialize fine-tuner
        fine_tuner = RobertoLoRAFineTuner()
        print("✅ LoRA fine-tuner initialized")

        # Test model setup (without training)
        fine_tuner.setup_model_and_tokenizer()
        print("✅ Model and tokenizer loaded with LoRA")

        # Check if LoRA was applied
        if hasattr(fine_tuner, 'peft_model') and fine_tuner.peft_model is not None:
            print("✅ LoRA adapters applied successfully")
            print(
                f"   Trainable parameters: {fine_tuner.peft_model.num_parameters()}")
        else:
            print("❌ LoRA adapters not applied")
            return False

        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 LoRA Implementation Test")
    print("=" * 50)

    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Install missing packages:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Test config
    if not test_config():
        print("\n❌ Config test failed. Check config.yaml")
        sys.exit(1)

    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ All tests passed! LoRA implementation is working correctly.")
    print("\nYou can now run:")
    print("   python fine_tuning_lora.py")
    print("   or")
    print("   python run_training.py")


if __name__ == "__main__":
    main()
