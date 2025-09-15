import os
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Config 

model_path = r"D:\AI-Inosuke\models\Qwen2.5-3B-Instruct"   
dataset_path = r"D:\AI-Inosuke\data\dataset.jsonl"
output_dir = r"D:\AI-Inosuke\models\inosuke-lora"
os.makedirs(output_dir, exist_ok=True)

# Hyperparams 
max_length = 512
per_device_batch = 1        # 1 sample per device
grad_accum = 16             # effective batch = per_device_batch * grad_accum
num_epochs = 6
learning_rate = 2e-4
warmup_ratio = 0.05
weight_decay = 0.01
r_lora = 64                 
lora_alpha = 16
logging_steps = 50

# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model (QLoRA via BitsAndBytesConfig)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading base model (quantized 4bit)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)


# Prepare model for k-bit + attach LoRA

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=r_lora,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("LoRA adapter attached.")

# Load & prepare dataset

print("Loading dataset...")
raw = load_dataset("json", data_files=dataset_path, split="train")

def format_example(example):
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    if inp and inp.strip():
        text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
    return {"text": text}

raw = raw.map(format_example, num_proc=1)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

print("Tokenizing dataset...")
tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])

# split train / val (90/10)
splits = tokenized.train_test_split(test_size=0.10, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]
print(f"Train samples: {len(train_dataset)}  |  Eval samples: {len(eval_dataset)}")

# Data collator

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# TrainingArguments 

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type="cosine",
    weight_decay=weight_decay,
    max_grad_norm=1.0,
    fp16=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=logging_steps,
    save_strategy="epoch",
    save_total_limit=3,
    eval_strategy="epoch",                 
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    optim="paged_adamw_32bit",
)


# Trainer (with EarlyStopping)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# Start training

print("Starting training... (use 'tensorboard --logdir=./logs' to monitor)")
trainer.train()

# Save final LoRA + tokenizer

print("Saving LoRA adapter + tokenizer to:", output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("FINISHED.")
