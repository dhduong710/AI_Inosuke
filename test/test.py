import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Config
BASE_MODEL = r"D:\AI-Inosuke\models\Qwen2.5-3B-Instruct"
LORA_ADAPTER = r"D:\AI-Inosuke\models\inosuke-lora\checkpoint-2605"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 4bit QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model 
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# Load LoRA 
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER)

# Chat function 
def chat_once(user_input: str, max_new_tokens=200):

    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    return decoded.strip()

# Chat
if __name__ == "__main__":
    print("=== Inosuke Chat (single turn) ===")
    while True:
        q = input("You: ")
        if q.strip().lower() in ["exit", "quit"]:
            break
        ans = chat_once(q)
        print("Inosuke:", ans)
