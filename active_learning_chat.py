import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
import sys
# Force UTF-8 output for Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from groq import Groq
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    AutoConfig
)
from trl import SFTTrainer
from datasets import Dataset
from memory_manager import QdrantMemory # Import Memory Manager

# Initialize Memory
memory = QdrantMemory()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_MODEL_DIR = "./phi3_base_model"
OUTPUT_DIR = "./phi3_self_improved"
DATA_FILE = "./data/self_improvement_data.json"

# API SETUP
groq_api_key = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
client = Groq(api_key=groq_api_key)

# ---------------------------------------------------------
# HELPER: Get Teacher Response
# ---------------------------------------------------------
def get_expert_answer(user_query):
    """
    Asks Llama-3 (Teacher) for the ideal answer.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful, detailed expert assistant. Provide comprehensive, in-depth, and well-explained answers."},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error contacting expert: {e}"

import gc

# ---------------------------------------------------------
# HELPER: Train Model
# ---------------------------------------------------------
def train_model(new_data):
    """
    Fine-tunes the model on the gathered data.
    """
    # Simply suppress output here or just print one line in main loop
    # print("\n🧠 System is entering Learning Mode...") 
    # print(f"📚 Digesting {len(new_data)} new knowledge chunks...")

    try:
        # Force cleanup before loading
        gc.collect()
        torch.cuda.empty_cache()

        # Load Model (Clean Slate or Previous Best)
        if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
            model_to_load = OUTPUT_DIR
        else:
            model_to_load = BASE_MODEL_DIR
        
        # ---------------------------------------------------------
        # Standard Float16 LoRA
        # ---------------------------------------------------------
        from peft import LoraConfig, get_peft_model, TaskType

        tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(model_to_load, trust_remote_code=False)
        config.attn_implementation = "eager"

        # Load Base Model in Float16
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False
        )
        
        model.config.use_cache = False 

        # 3. LoRA Adapter Config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16,           # Rank (Higher = more trainable params, but slower)
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all Linear layers for best results
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters() # Verify the % is small (but powerful)

        # ---------------------------------------------------------
        # TRAINER
        # ---------------------------------------------------------
        from transformers import Trainer, DataCollatorForLanguageModeling

        # 1. Prepare Dataset
        dataset_list = []
        for item in new_data:
            # CLEANER PROMPT FORMAT
            full_text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}<|endoftext|>"
            dataset_list.append({"text": full_text})
        
        raw_dataset = Dataset.from_list(dataset_list)

        # 2. Tokenize
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=512, 
                padding="max_length"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

        # 3. Training Arguments 
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=1e-4, 
            fp16=True,
            gradient_checkpointing=False, 
            logging_steps=10,
            save_strategy="no", 
            optim="adamw_torch", 
            report_to="none"
        )

        # 4. Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        print(f"   (Training on {len(tokenized_datasets)} examples...)")
        trainer.train()
        
        print("   ✅ Training complete!")
        print("   💾 Saving new intelligence (Adapters only)...")
        model.save_pretrained(OUTPUT_DIR) 
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Clean up memory
        del model
        del trainer
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Training Successful.")

    except Exception as e:
        print(f"❌ Training Failed: {e}")
        # Attempt cleanup again
        try:
            del model
            del trainer
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main():
    # Load existing data history/memory
    history = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            history = json.load(f)

    # Create a set for fast lookup of existing questions
    existing_instructions = set()
    for item in history:
        if "instruction" in item:
            existing_instructions.add(item["instruction"])

    session_data = [] # Data collected JUST in this session

    AUTO_TRAIN_THRESHOLD = 1 # <--- CHANGE THIS NUMBER to train more/less often

    print("\n\U0001f916 ACTIVE LEARNING BOT INITIALIZED")
    print("-----------------------------------")
    print(f"Auto-Train set to every {AUTO_TRAIN_THRESHOLD} messages.")
    print("Type 'quit' to exit.")
    print("-----------------------------------\n")

    while True:
        # Check for Auto-Train
        if len(session_data) >= AUTO_TRAIN_THRESHOLD:
            print(f"\n\u26a1 Auto-Training ({len(session_data)} items)...")
            train_model(session_data)
            
            # Update local memory
            history.extend(session_data)
            for item in session_data:
                existing_instructions.add(item["instruction"])
                
            with open(DATA_FILE, "w") as f:
                json.dump(history, f, indent=4)
            
            session_data = [] # Reset
            print("Chat resumed.")
            print("-----------------------------------")

        try:
            user_input = input("\U0001f464 You: ").strip()
        except EOFError:
            break
            
        if not user_input: # Skip empty inputs
            continue

        if user_input.lower() in ["quit", "exit"]:
            break
            
        if user_input.lower() == "train":
            if len(session_data) == 0:
                print("\u26a0\ufe0f Nothing to train.")
                continue
            
            print(f"\n\u26a1 Manual Training ({len(session_data)} items)...")
            train_model(session_data)
            
            # Update local memory
            history.extend(session_data)
            for item in session_data:
                existing_instructions.add(item["instruction"])
                
            with open(DATA_FILE, "w") as f:
                json.dump(history, f, indent=4)
                
            session_data = [] # Reset buffer
            print("Chat resumed.")
            continue

        # 1. Get Expert Answer
        expert_answer = get_expert_answer(user_input)
        
        print(f"\U0001f916 Expert Answer: {expert_answer}")
        
        # 2. Store Data (Only if unique)
        if user_input not in existing_instructions:
            # Check if it's already in the current session too
            if not any(d['instruction'] == user_input for d in session_data):
                new_entry = {
                    "instruction": user_input,
                    "input": "",
                    "output": expert_answer
                }
                session_data.append(new_entry)
                
                # --- NEW: Save to Vector Memory (Chunks) ---
                print("   [Saving to Vector Memory...]")
                memory.add_memory(expert_answer, metadata={"instruction": user_input})
                # -------------------------------------------
                
                print(f"   [Added to Learning Queue: {len(session_data)}/{AUTO_TRAIN_THRESHOLD}]")
            else:
                print(f"   [Note: Already in current learning queue.]")
        else:
            print(f"   [Note: Already learned this previously. Skipping storage.]")


if __name__ == "__main__":
    main()
