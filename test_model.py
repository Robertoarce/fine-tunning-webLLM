#!/usr/bin/env python3
"""
Test script for the fine-tuned Roberto Arce model
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text using the model"""
    if model is None or tokenizer is None:
        print("❌ Model not loaded!")
        return None

    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    """Main testing function"""
    print("🧪 Testing Fine-tuned Roberto Arce Model")
    print("=" * 50)

    # Find the latest model directory
    model_dirs = [d for d in os.listdir('.') if d.startswith(
        'finetuned_roberto') and os.path.isdir(d)]

    if not model_dirs:
        print("❌ No fine-tuned model found!")
        print("   Please run the training script first.")
        return

    # Use the most recent model (assuming timestamp format)
    latest_model = sorted(model_dirs)[-1]
    print(f"📁 Using model: {latest_model}")

    # Load model
    model, tokenizer = load_model(latest_model)
    if model is None:
        return

    # Test prompts
    test_prompts = [
        "Roberto Arce is a",
        "Roberto's expertise includes",
        "Roberto studied",
        "Roberto works as a",
        "Roberto's professional experience",
        "Roberto's education background",
        "Roberto specializes in",
        "Roberto's skills include"
    ]

    print("\n🎯 Testing with various prompts:")
    print("=" * 50)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        print("-" * 30)

        generated = generate_text(model, tokenizer, prompt)
        if generated:
            # Remove the original prompt from the output for cleaner display
            response = generated.replace(prompt, "").strip()
            print(f"Response: {response}")
        else:
            print("❌ Failed to generate text")

    # Interactive mode
    print("\n" + "=" * 50)
    print("🎮 Interactive Mode - Enter your own prompts!")
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        try:
            user_prompt = input("\nEnter prompt: ").strip()
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                break

            if not user_prompt:
                continue

            generated = generate_text(model, tokenizer, user_prompt)
            if generated:
                response = generated.replace(user_prompt, "").strip()
                print(f"Response: {response}")
            else:
                print("❌ Failed to generate text")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
