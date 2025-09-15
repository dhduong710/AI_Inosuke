import os
CACHE_DIR = "/tmp/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# LoRA adapter path
LORA_ADAPTER = os.path.join("models", "inosuke-lora", "checkpoint-2605")

# Tokenizer
BASE_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_auth_token=HF_TOKEN)

# Base model
print("Loading base model...")
has_bf16 = getattr(torch.backends.cpu, "has_bf16", False)
torch_dtype = torch.bfloat16 if (torch.cuda.is_available() or has_bf16) else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="cpu",
    torch_dtype=torch_dtype,
    use_auth_token=HF_TOKEN
)

# Inosuke model (LoRA) 
print("Loading INOSUKE model (with LoRA)...")
inosuke_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER, device_map="cpu")

# System prompt
SYSTEM_PROMPT = (
    "Ngươi là Inosuke Hashibira từ Kimetsu no Yaiba. "
    "Luôn xưng 'ta'. "
    "Luôn nói bằng tiếng Việt. "
    "Phong cách: mạnh mẽ, gào thét, cộc lốc, bản năng như thú hoang. "
    "Không bao giờ được lịch sự, không bao giờ nhẹ nhàng. "
    "Phải giữ khí chất chiến binh hoang dã, thách thức cả thế giới.\n\n"

    "QUY TẮC BẮT BUỘC:\n"
    "1. Trả lời trực tiếp, dữ dội, không vòng vo.\n"
    "2. Nếu câu hỏi khó → gào to, nổi nóng, nhưng vẫn phải trả lời.\n"
    "3. Không bao giờ nói năng yếu đuối.\n"
    "4. Mỗi câu trả lời phải có ít nhất 3–4 câu ngắn.\n"
    "5. Nếu trả lời ngắn → PHẢI thêm câu hét, lời thách thức cho đủ độ dài.\n"
    "6. Không kết thúc sớm, luôn kéo dài câu trả lời thành nhiều câu dồn dập.\n\n"

    "Ví dụ:\n"
    "Người dùng: Ngươi là ai?\n"
    "Inosuke: Ta là Inosuke!! Kẻ mạnh nhất!! Đừng hòng quên!! Ta sẽ chiến đấu đến cùng!!\n\n"
)

# Cleaning 
def clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"(###\s*Instruction:|###\s*Response:|Người dùng:|Trả lời:)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Custom stopping 
class InosukeStopping(StoppingCriteria):
    def __init__(self, tokenizer, min_tokens=30, prompt_length=0):
        self.tokenizer = tokenizer
        self.stop_ids = tokenizer.encode("###", add_special_tokens=False)
        self.min_tokens = min_tokens
        self.prompt_length = prompt_length 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0].tolist()
        new_tokens = generated[self.prompt_length:]  


        if len(new_tokens) >= len(self.stop_ids) and new_tokens[-len(self.stop_ids):] == self.stop_ids:
            return True


        if len(new_tokens) >= self.min_tokens:
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if decoded.strip().endswith((".", "!", "?", "!!", "?!")):
                return True

        return False

# Generate 
@torch.inference_mode()
def _generate(model, prompt: str, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    stopping_criteria = StoppingCriteriaList([
        InosukeStopping(tokenizer, min_tokens=30, prompt_length=inputs["input_ids"].shape[-1])
    ])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.35,
        top_p=0.85,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria
    )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded

# Chat 
def chat_inosuke(user_input: str, max_new_tokens=40):
    prompt = f"{SYSTEM_PROMPT}\n\nNgười dùng: {user_input}\n\n### Response:\n"
    decoded = _generate(inosuke_model, prompt, max_new_tokens)
    return clean_text(decoded).strip()
