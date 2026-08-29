from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, PrefixTuningConfig, TaskType, PeftModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

total_params = sum(p.numel() for p in model.parameters())
print(f"Full GPT-2 parameters: {total_params:,}")

peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

examples = [
    "MR: name[The Eagle] food[Chinese] priceRange[cheap] SENT: The Eagle is a cheap Chinese restaurant.",
    "MR: name[Zeta] food[Italian] priceRange[moderate] SENT: Zeta is a moderately priced Italian restaurant.",
]
enc = tokenizer(examples, return_tensors="pt", padding=True)
labels = enc["input_ids"].clone()

optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-3)
peft_model.train()
for step in range(10):
    out = peft_model(**enc, labels=labels)
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
    print(f"step {step}: loss = {out.loss.item():.4f}")

peft_model.save_pretrained("prefix_adapter")
reloaded_base = GPT2LMHeadModel.from_pretrained("gpt2")
reloaded = PeftModel.from_pretrained(reloaded_base, "prefix_adapter")
print("Saved and reloaded successfully.")