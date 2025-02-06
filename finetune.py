from datasets import load_dataset
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from transformers import TrainingArguments

# local_rank = os.getenv("LOCAL_RANK")
os.environ["WANDB_DISABLED"] = "true"
# device_string = "cuda:0"# + str(local_rank)

# Load the dataset
# dataset_name = "charlieoneill/hypothesis_generation"
# dataset = load_dataset(dataset_name, split="train")

alpaca_prompt = """Below is a scientific question framed as a "bit-spark-flip". A bit is a conventional belief or problem in computer science, a "spark" is a 4-6 word seed of an idea that could be a solution/hypothesis to this problem, and the "flip" is an expansion on this innovative approach or hypothesis (spark) that challenges this bit.

### Bit:
{}

### Spark:
{}

### Flip:
{}"""

def formatting_prompts_func(examples):
    bit_flips = examples["bit_flip_spark"]
    texts = []
    for bf in bit_flips:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(bf['bit'], bf['spark'], bf['flip'])
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="neurips2023_reasoning.json", split='train')
dataset = dataset.map(formatting_prompts_func, batched = True)


# Load the model + tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache = True,
    device_map="auto", #{device_string}
)
exit(0)

# PEFT config
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
)


# Args 
max_seq_length = 512
output_dir = "./results"
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
optim = "adamw_hf"
save_steps = 1000
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 5763
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=False, # Otherwise weird loss pattern (see https://github.com/artidoro/qlora/issues/84#issuecomment-1572408347, https://github.com/artidoro/qlora/issues/228, https://wandb.ai/answerdotai/fsdp_qlora/runs/snhj0eyh)
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False}, # Needed for DDP
    # report_to="wandb",
    report_to=None,
)

# Trainer 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Not sure if needed but noticed this in https://colab.research.google.com/drive/1t3exfAVLQo4oKIopQT1SKxK4UcYg7rC1#scrollTo=7OyIvEx7b1GT
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Train :)
trainer.train()

model = AutoModelForCausalLM.from_pretrained(
    "/scratch/x77/cn1951/llama3",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# # Load our fine-tuned model
# path_to_model_folder = "results/checkpoint-4000"
# model = PeftModel.from_pretrained(model, path_to_model_folder)
# model = model.merge_and_unload()
# model.push_to_hub(repo_id="charlieoneill/llama3-8b-hypogen"