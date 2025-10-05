# Integrating RAFT Model into Your Website

## Overview

You currently have:

- ✅ Vue.js website with chatbot
- ✅ Basic RAG system (keyword search)
- ✅ Base LLM (`Xenova/distilgpt2`)
- ⚠️ Not trained → hallucination issues

You have **two main options** to improve your chatbot:

---

## Option 1: Use RAFT Fine-tuned Model (Best Quality) 🌟

### Pros:

- ✅ **Best answer quality** - Trained on Roberto's data
- ✅ **Minimal hallucination** - Trained to extract from context
- ✅ **Context-aware** - RAFT methodology
- ✅ **Already trained** - Just need to deploy

### Cons:

- ❌ Requires backend server (can't run in browser)
- ❌ More complex setup
- ❌ Hosting costs (but can use free tier initially)

### Architecture:

```
User Question (Browser)
        ↓
Frontend RAG System (ragSystem.js)
├─ Search roberto.txt & repositories
├─ Find top 5 relevant chunks
└─ Format as RAFT prompt
        ↓
HTTP Request to Backend
        ↓
Backend Server (FastAPI/Flask)
├─ Load RAFT fine-tuned model
├─ Generate answer from prompt
└─ Return response
        ↓
Display in Chatbot
```

### Setup Steps:

#### 1. Create Backend API (Python FastAPI)

```python
# backend/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Enable CORS for your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-website.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAFT model once at startup
MODEL_PATH = "./finetuned_roberto_raft_20250103_120000"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.post("/generate")
async def generate(request: QueryRequest):
    inputs = tokenizer.encode(request.prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + request.max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.95
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response[len(request.prompt):].strip()

    return {"response": answer}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

#### 2. Update Frontend `localLLM.js`

```javascript
// src/utils/localLLM.js
import { ragSystem } from "./ragSystem.js";

class LocalLLM {
  constructor() {
    this.apiUrl = "https://your-backend-api.com"; // Your deployed backend
    // For local dev: 'http://localhost:8000'
  }

  async generateResponse(userQuery) {
    try {
      // Generate context from RAG system
      await ragSystem.initialize();
      const context = await ragSystem.generateContext(userQuery);

      // Create RAFT-style prompt
      const prompt = `${context}

Question: ${userQuery}
Answer:`;

      // Call backend API
      const response = await fetch(`${this.apiUrl}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
          max_tokens: 150,
        }),
      });

      const data = await response.json();
      return data.response;
    } catch (error) {
      console.error("Error generating response:", error);
      return "I'm sorry, I'm having trouble processing your request right now.";
    }
  }
}

export const localLLM = new LocalLLM();
```

#### 3. Deploy Backend

**Option A: Render.com (Free tier)**

```bash
# Install dependencies
pip install fastapi uvicorn transformers torch

# Run locally first
uvicorn api:app --reload

# Deploy to Render
# 1. Push to GitHub
# 2. Create Web Service on Render
# 3. Connect your GitHub repo
# 4. Build command: pip install -r requirements.txt
# 5. Start command: uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Option B: Railway.app (Free tier)**

- Push code to GitHub
- Connect to Railway
- Auto-deploys

**Option C: Hugging Face Spaces (Free)**

```python
# app.py for Hugging Face Spaces
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", "your-username/roberto-raft-model")

def generate(prompt):
    result = pipe(prompt, max_new_tokens=150)
    return result[0]['generated_text']

iface = gr.Interface(fn=generate, inputs="text", outputs="text")
iface.launch()
```

---

## Option 2: Fine-tune Browser Model (Simpler Setup) ⚡

If you want to keep everything in the browser (no backend needed), you can fine-tune a smaller model that runs client-side.

### Pros:

- ✅ **No backend needed** - Runs in browser
- ✅ **Free hosting** - Static site
- ✅ **Fast deployment** - Just update files
- ✅ **Privacy** - All processing client-side

### Cons:

- ❌ **Limited model size** - Only small models work
- ❌ **Slower inference** - Browser is slower than server
- ❌ **Still some hallucination** - Smaller models less accurate

### Models That Work in Browser:

| Model               | Size  | Quality  | Speed  |
| ------------------- | ----- | -------- | ------ |
| `Xenova/distilgpt2` | 82MB  | ⭐⭐     | Fast   |
| `Xenova/gpt2`       | 240MB | ⭐⭐⭐   | Medium |
| `Xenova/Phi-1_5`    | 1.6GB | ⭐⭐⭐⭐ | Slower |

### Implementation:

#### 1. Fine-tune Smaller Model

Use your RAFT training data but with a smaller model:

```python
# In /13 RAFT directory
# Edit config.yaml
model:
  name: "distilgpt2"  # Already using this

# Run RAFT training
python run_raft.py
```

#### 2. Convert to ONNX for Browser

```python
# convert_to_onnx.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnx import export

model_path = "./finetuned_roberto_raft_20250103_120000"
output_path = "./onnx_model"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Convert to ONNX
export(
    model=model,
    tokenizer=tokenizer,
    output_path=output_path,
    task="text-generation"
)

print(f"Model converted to ONNX: {output_path}")
```

#### 3. Upload to Hugging Face

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload your model
huggingface-cli upload your-username/roberto-raft-onnx ./onnx_model
```

#### 4. Use in Browser

```javascript
// Update localLLM.js
import { pipeline } from "@xenova/transformers";

class LocalLLM {
  constructor() {
    this.pipe = null;
    this.config = {
      modelName: "your-username/roberto-raft-onnx", // Your uploaded model
      quantized: true,
      // ... rest of config
    };
  }

  // Rest stays the same!
}
```

---

## Comparison: Which Option Should You Choose?

| Aspect               | Option 1: Backend API    | Option 2: Browser Model |
| -------------------- | ------------------------ | ----------------------- |
| **Setup Complexity** | Medium-High              | Low-Medium              |
| **Answer Quality**   | ⭐⭐⭐⭐⭐ Excellent     | ⭐⭐⭐ Good             |
| **Speed**            | Fast (GPU server)        | Slower (CPU browser)    |
| **Cost**             | Free tier available      | Completely free         |
| **Maintenance**      | Need to maintain server  | Static files only       |
| **Privacy**          | Data sent to server      | Everything client-side  |
| **Model Size**       | Any size (GPT-2 Medium+) | Limited (<500MB)        |
| **Hallucination**    | Minimal                  | Low-Medium              |

---

## My Recommendation 🎯

### For Best Results: **Hybrid Approach**

1. **Deploy RAFT backend API** (Option 1)
   - Use for production
   - Best quality answers
2. **Keep browser fallback** (Option 2)
   - Use if API fails
   - Works offline
   - Better than nothing

### Implementation:

```javascript
// src/utils/localLLM.js
class LocalLLM {
  constructor() {
    this.apiUrl = "https://your-api.com";
    this.browserPipe = null;
    this.useBackend = true; // Try backend first
  }

  async generateResponse(userQuery) {
    // Try backend API first
    if (this.useBackend) {
      try {
        const response = await this.generateFromBackend(userQuery);
        return response;
      } catch (error) {
        console.warn("Backend failed, falling back to browser model");
        this.useBackend = false;
      }
    }

    // Fallback to browser model
    return await this.generateFromBrowser(userQuery);
  }

  async generateFromBackend(userQuery) {
    // ... backend implementation
  }

  async generateFromBrowser(userQuery) {
    // ... existing browser implementation
  }
}
```

---

## Quick Start Steps

### Immediate (Use Browser Model Better)

1. Your current setup is fine
2. Just add better prompting in `ragSystem.js`:

```javascript
async generateContext(query) {
  // ... existing search code ...

  context += "\nYou are Roberto Arce's AI assistant. CRITICAL RULES:\n";
  context += "1. ONLY answer based on the information above\n";
  context += "2. If information is not above, say 'I don't have that information'\n";
  context += "3. Never make up information\n";
  context += "4. Be concise and factual\n";
  context += "5. When unsure, ask for clarification\n";

  return context;
}
```

### This Week (Train & Deploy RAFT)

1. Train RAFT model (already done in `/13 RAFT`)
2. Create FastAPI backend
3. Deploy to Render/Railway (free tier)
4. Update frontend to use API
5. Remove WIP warning! 🎉

### Next Month (Production Ready)

1. Add caching
2. Rate limiting
3. Error handling
4. Monitoring
5. A/B testing

---

## Need Help?

**Which option interests you more?**

- Option 1 (Backend API) → I'll help you set it up
- Option 2 (Browser only) → I'll optimize your current setup
- Hybrid → Best of both worlds

Let me know and I'll create the specific files you need! 🚀
