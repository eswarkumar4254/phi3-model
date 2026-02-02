import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import sys
import json
# Force UTF-8 output for Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from memory_manager import QdrantMemory

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Logic: Try loading the "self_improved" model first. 
# If not found, load the "base" model.
IMPROVED_MODEL_DIR = "./phi3_self_improved"
BASE_MODEL_DIR = "./phi3_base_model"
DATA_FILE = "./data/self_improvement_data.json"

# Load knowledge base for "Retrieval"
knowledge_base = {}
if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, "r", encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                knowledge_base[item["instruction"].strip().lower()] = item["output"]
    except:
        pass

# 1. Initialize Memory
memory = QdrantMemory()

# ---------------------------------------------------------
# LOAD MODEL (Base + Adapter)
# ---------------------------------------------------------
from peft import PeftModel

# Always load the BASE model first
print(f"⏳ Loading Base Model: {BASE_MODEL_DIR}...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=False)
tokenizer.pad_token = tokenizer.eos_token 

config = AutoConfig.from_pretrained(BASE_MODEL_DIR, trust_remote_code=False)
config.attn_implementation = "eager"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False
)

# Check if we have a fine-tuned adapter
if os.path.exists(IMPROVED_MODEL_DIR) and os.path.exists(os.path.join(IMPROVED_MODEL_DIR, "adapter_config.json")):
    print(f"✅ Found Fine-Tuned Adapter: {IMPROVED_MODEL_DIR}")
    print("   Merging Adapter into Brain...")
    model = PeftModel.from_pretrained(base_model, IMPROVED_MODEL_DIR)
else:
    print("⚠️ No Fine-Tuned Adapter found. Using Base Brain.")
    model = base_model

print("\n\U0001f916 LOCAL CHAT BOT (Offline Mode)")
print("-----------------------------------")
print("This bot uses ONLY your local model's brain.")
print("Type 'quit' to exit.")
print("-----------------------------------\n")

while True:
    try:
        user_input = input("\U0001f464 You: ").strip()
    except EOFError:
        break
        
    if not user_input:
        continue
        
    if user_input.lower() in ["quit", "exit"]:
        break

    # 1. SMART RETRIEVAL: Check Vector Memory
    # -------------------------------------------------------------
    retrieved_chunks = memory.search(user_input, limit=1)
    
    context_text = ""
    if retrieved_chunks:
        print(f"   \U0001f50d Used Memory Context: {len(retrieved_chunks)} chunk(s)")
        context_text = "\n".join(retrieved_chunks)
        # Optional: Print the memory it found
        # print(f"   [Context]: {context_text[:100]}...") 

    # 2. GENERATE ANSWER 
    # -------------------------------------------------------------
    # Format input with Context + Instruction
    if context_text:
        prompt = f"### Context:\n{context_text}\n\n### Instruction:\n{user_input}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=True,      # Enabled sampling with stability
            temperature=0.8,     # Balanced creativity
            top_p=0.9,           # Filtering
            repetition_penalty=1.2, # Prevent looping
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode 
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Robust Extraction
    if "### Response:" in response_text:
        final_answer = response_text.split("### Response:")[1].strip()
    else:
        # If extraction fails, show everything after the instruction
        final_answer = response_text.replace(prompt, "").strip() # Better clean up than before
        
    if not final_answer:
        final_answer = "(Model generated an empty response. It may need more training time.)"

    print(f"\U0001f916 AI: {final_answer}")
    print("-" * 20)
