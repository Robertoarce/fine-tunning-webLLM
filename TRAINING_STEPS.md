# 🚀 RAFT Model Training - Step by Step

## 📋 Pre-Training Checklist

Before we start, make sure you have:

- ✅ Python 3.8+ installed
- ✅ Your training data (`roberto_data.txt`)
- ✅ Enough disk space (~2GB for model)
- ✅ Time: 15-60 minutes depending on CPU/GPU

---

## 🎯 Training Process (3 Steps)

### Step 1: Install Dependencies

```bash
cd "/Users/i0557807/00 ALL/02 Me/13 RAFT"

# Install all requirements
pip install -r requirements.txt
```

**Expected output:**

```
Successfully installed torch transformers datasets accelerate ...
```

---

### Step 2: Check Your Configuration

Open `config.yaml` and verify settings:

```yaml
model:
  name: "distilgpt2" # Good for CPU/Mac
  max_length: 512
  fp16: false # Set to true if you have GPU

training:
  num_epochs: 5 # 5 is good for start
  batch_size: 8 # Reduce to 4 if out of memory
  learning_rate: 5e-5
```

**Adjust if needed:**

- 💻 **CPU only:** `batch_size: 4`, `fp16: false`
- 🚀 **GPU:** `batch_size: 8`, `fp16: true`
- ⚡ **Fast test:** `num_epochs: 2`

---

### Step 3: Run Training! 🎉

```bash
# Complete pipeline (data generation + training)
python run_raft.py
```

**What will happen:**

```
================================================================================
RAFT (Retrieval Augmented Fine Tuning) Pipeline
================================================================================

[STEP 1] Generating RAFT Training Dataset...
--------------------------------------------------------------------------------
Loading documents from roberto_data.txt
Created 45 document chunks
Generating RAFT dataset...
Generated 234 RAFT training examples
✓ RAFT dataset generated successfully: raft_training_data.jsonl

[STEP 2] Fine-tuning Model with RAFT...
--------------------------------------------------------------------------------
Loading configuration from config.yaml
Loading model: distilgpt2
✓ Tokenizer loaded
✓ Model loaded successfully!
Preparing RAFT datasets...
✓ Loaded 234 training examples
Training samples: 210
Validation samples: 24

Starting training...
Epoch 1/5: 100%|██████████| 27/27 [05:23<00:00]
  train_loss: 2.345
  eval_loss: 1.987

Epoch 2/5: 100%|██████████| 27/27 [05:18<00:00]
  train_loss: 1.567
  eval_loss: 1.432

...

✓ Model fine-tuned successfully: ./finetuned_roberto_raft_20250105_153045

[STEP 3] Testing Model...
✓ Training completed!
```

---

## ⏱️ Expected Training Time

| Hardware          | Time          |
| ----------------- | ------------- |
| 🍎 **Mac M1/M2**  | 15-25 minutes |
| 💻 **Modern CPU** | 30-45 minutes |
| 🖥️ **Older CPU**  | 45-90 minutes |
| 🚀 **GPU (CUDA)** | 5-10 minutes  |

---

## 🔍 Monitoring Progress

While training, you'll see:

```
{'loss': 2.234, 'learning_rate': 4.5e-05, 'epoch': 0.5}
{'loss': 1.876, 'learning_rate': 4.0e-05, 'epoch': 1.0}
{'eval_loss': 1.543, 'eval_accuracy': 0.82, 'epoch': 1.0}
```

**Good signs:**

- ✅ `loss` decreasing (2.0 → 1.5 → 1.0)
- ✅ `eval_loss` staying stable
- ✅ No errors

**Warning signs:**

- ⚠️ `loss` increasing
- ⚠️ Out of memory errors
- ⚠️ `eval_loss` much higher than `loss`

---

## 🛠️ Troubleshooting

### Out of Memory

Edit `config.yaml`:

```yaml
training:
  batch_size: 2 # Reduce from 8
  gradient_accumulation_steps: 4 # Increase from 2
```

### Training Too Slow

```yaml
training:
  num_epochs: 3 # Reduce from 5
model:
  max_length: 256 # Reduce from 512
```

### Model Not Learning (loss not decreasing)

```yaml
training:
  learning_rate: 1e-4 # Increase from 5e-5
  num_epochs: 7 # Train longer
```

---

## ✅ Success Indicators

After training completes, you should see:

1. **New folder created:**

   ```
   finetuned_roberto_raft_20250105_153045/
   ├── config.json
   ├── pytorch_model.bin (or model.safetensors)
   ├── tokenizer_config.json
   └── training_log.json
   ```

2. **Test results showing:**

   ```
   Question: Where does Roberto work?
   Answer: Based on the documents, Roberto works at Sanofi as a Data Scientist...
   ```

3. **Final eval_loss < 1.5**

---

## 🎉 After Training

Your model is ready! Now:

1. **Test it:**

   ```bash
   python test_raft_model.py
   ```

2. **Use in backend:**

   - Model path is automatically found
   - Start backend: `./start-backend.sh`

3. **Deploy to website:**
   - Backend picks up latest model
   - Frontend connects automatically

---

## 📊 Training Statistics

Check `training_log.json` in model folder:

```json
{
  "epoch": 5.0,
  "train_loss": 1.234,
  "eval_loss": 1.456,
  "train_samples": 210,
  "eval_samples": 24
}
```

**What the numbers mean:**

- **train_loss:** How well model fits training data (lower = better)
- **eval_loss:** How well model generalizes (should be close to train_loss)
- **Gap > 0.5:** Might be overfitting (train longer with more data)

---

## 🎓 Advanced Options

### Train with Different Model

```yaml
model:
  name: "gpt2" # Larger, better quality (slower)
  # or "gpt2-medium" (even better, much slower)
```

### Customize Data Generation

Edit `raft_data_generator.py`:

```python
generator = RAFTDataGenerator(
    chunk_size=200,          # Smaller chunks (more examples)
    num_distractors=5,       # More distractors (harder training)
    distractor_probability=0.7  # 70% with distractors
)
```

### More Epochs

```yaml
training:
  num_epochs: 10 # Train longer for better quality
```

---

## 🚀 Quick Commands Reference

```bash
# Complete pipeline
python run_raft.py

# Just generate data
python raft_data_generator.py

# Just train (if data exists)
python raft_fine_tuning.py

# Test model
python test_raft_model.py

# Test with specific model
python test_raft_model.py ./finetuned_roberto_raft_20250105_153045
```

---

## 📞 Need Help?

**Common issues:**

- Memory error → Reduce batch_size
- Too slow → Reduce epochs or use smaller model
- Poor quality → Train longer, check data

**Check logs:**

```bash
# View last training
tail -f *.log
```

---

Ready to train? Just run:

```bash
cd "/Users/i0557807/00 ALL/02 Me/13 RAFT"
python run_raft.py
```

And watch the magic happen! ✨
