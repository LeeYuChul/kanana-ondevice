import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "kakaocorp/kanana-nano-2.1b-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "안녕,너의 이름은 뭐고 어떤 역할을 할 수 있어?"
messages = [
    {"role": "system", "content": "You are a helpful AI assistant developed by Kakao."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt").to(device)

_ = model.eval()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=2000,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_tokens = output[0, inputs["input_ids"].shape[-1] :]
print(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())

# Output:
# Sure! Here are the given dates converted to the `YYYY/MM/DD` format:

# 1. **12/31/2021**
#    - **YYYY/MM/DD:** 2021/12/31

# 2. **02-01-22**
#    - **YYYY/MM/DD:** 2022/02/01

# So, the converted dates are ...
