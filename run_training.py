#!/usr/bin/env python3
"""
Simple script to run Roberto Arce fine-tuning
Run this after installing requirements: pip install -r requirements.txt
"""

import sys
import os


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'datasets', 'accelerate',
        'scikit-learn', 'numpy', 'yaml'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with: pip install -r requirements.txt")
        return False

    print("✅ All required packages are installed!")
    return True


def main():
    """Main function"""
    print("🚀 Starting Roberto Arce Fine-tuning")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check if data file exists
    if not os.path.exists("roberto_data.txt"):
        print("❌ roberto_data.txt not found!")
        print("   Make sure the training data file is in the current directory.")
        sys.exit(1)

    print("✅ Training data found!")

    # Ask user which script to run
    print("\nChoose training script:")
    print("1. Basic fine-tuning (fine-tunning.py)")
    print("2. Advanced fine-tuning (fine_tuning_advanced.py) - Recommended")
    print("3. LoRA fine-tuning (fine_tuning_lora.py) - Memory Efficient")

    choice = input("\nEnter choice (1, 2, or 3): ").strip()

    if choice == "1":
        print("\n🔄 Running basic fine-tuning...")
        os.system("python fine-tunning.py")
    elif choice == "2":
        print("\n🔄 Running advanced fine-tuning...")
        os.system("python fine_tuning_advanced.py")
    elif choice == "3":
        print("\n🔄 Running LoRA fine-tuning...")
        os.system("python fine_tuning_lora.py")
    else:
        print("❌ Invalid choice. Please run the script again.")
        sys.exit(1)

    print("\n✅ Training completed!")
    print("📁 Check the output directory for your fine-tuned model.")


if __name__ == "__main__":
    main()
