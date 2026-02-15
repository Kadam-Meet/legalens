#!/usr/bin/env python3
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers.trainer_utils import set_seed

# ---------------- CONFIG ----------------
set_seed(42)

MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_GLOB = "training_json/*.json"

# Load HF token from environment variable for security
HF_TOKEN = os.getenv('HF_TOKEN', '')

OUTPUT_DIR = "training_output"
FINAL_MODEL_DIR = "models/mistral_full"   # ← FULL offline model

MAX_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4
MAX_EPOCHS = 5

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

assert torch.cuda.is_available(), "❌ GPU NOT DETECTED"
print("✅ GPU:", torch.cuda.get_device_name())

# ---------------- PROMPT FORMAT ----------------
def build_prompt(example):
    return {
        "text": f"""You are a legal AI assistant.

Instruction:
{example['instruction']}

Context:
{example['context']}

Answer:
{example['output']}
""".strip()
    }

# ---------------- EARLY STOPPING ----------------
class LossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=50, min_delta=0.01, stop_loss=0.85):
        self.patience = patience
        self.min_delta = min_delta
        self.stop_loss = stop_loss
        self.best_loss = None
        self.counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        loss = logs["loss"]

        if loss <= self.stop_loss:
            print(f"\n🛑 Early stopping (loss {loss:.4f})")
            control.should_training_stop = True
            return

        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("\n🛑 Early stopping (plateau)")
            control.should_training_stop = True

# ---------------- MAIN ----------------
def main():
    print("📦 Loading dataset...")
    dataset = load_dataset("json", data_files={"train": DATA_GLOB}, split="train")
    dataset = dataset.map(build_prompt, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length"
        )

    dataset = dataset.map(tokenize, batched=True)

    # ---- 4-bit NF4 ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print("🧠 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb,
        token=HF_TOKEN
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA ----
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            num_train_epochs=MAX_EPOCHS,
            fp16=True,
            logging_steps=1,
            save_steps=200,
            save_total_limit=2,
            report_to="none",
            optim="paged_adamw_8bit"
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[LossEarlyStoppingCallback()]
    )

    print("🚀 Training started...")
    trainer.train()

    # ---------------- MERGE LoRA → BASE ----------------
    print("🧩 Merging LoRA into base model...")
    model = model.merge_and_unload()

    print("💾 Saving FULL model...")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("✅ Offline model saved at:", FINAL_MODEL_DIR)

if __name__ == "__main__":
    main()
