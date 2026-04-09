import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "kakaocorp/kanana-nano-2.1b-base"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

prompt1 = "이처럼 인간처럼 생각하고 행동하는 AI 모델은 "
prompt2 = "Kakao is a leading company in South Korea, and it is known for "

input_ids = tokenizer(
    [prompt1, prompt2],
    padding=True,
    return_tensors="pt",
)["input_ids"].to("cuda")

_ = model.eval()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=32,
        do_sample=False,
    )

decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
for text in decoded:
    print(text)

# Output:
# 이처럼 인간처럼 생각하고 행동하는 AI 모델은 2020년대 중반에 등장할 것으로 예상된다. 2020년대 중반에 등장할 것으로 예상되는 AI 모델은 인간
# Kakao is a leading company in South Korea, and it is known for 1) its innovative products and services, 2) its commitment to sustainability, and 3) its focus on customer experience. Kakao has been recognized as
