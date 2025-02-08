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

alpaca_prompt = """Below is a scientific question framed as a "bit-spark-flip". You will be given a Bit, which is a statement of a conventional belief or a recognized problem in computer science. Based on this Bit, your task is to generate:
1. Spark (4-6 words): A succinct, innovative idea or hypothesis that addresses or challenges the Bit.
2. Flip (2-3 sentences): An expanded explanation of how this new idea disrupts or rethinks the assumptions in the Bit.
3. Reasoning Chain (multi-step): A detailed, step-by-step explanation referencing how we move from the conventional Bit to the innovative Flip, using any relevant details or evidence that might have been in the original text.
Your output should follow this format:
Spark: <4-6 word innovative seed>
Flip: <2-3 sentences explaining how the Spark challenges the Bit>
Reasoning Chain: <multi-step explanation showing how the Bit leads to the Flip>
Example Input (Bit):
“Transformer-based models are often used for scientific hypothesis generation, but they rely on many parameters to handle long sequences. This reliance on large models creates efficiency and accessibility challenges for researchers and practitioners.”
Desired Output Structure:
Spark: Parameter-efficient state space hypothesis generation

Flip: Instead, State Space Models like Mamba can manage very long sequences with fewer parameters. This approach offers a more parameter-efficient way to tackle hypothesis generation without compromising performance.


Reasoning Chain: 
I started by recognizing that large language models, particularly transformer-based architectures, have become a common choice for scientific hypothesis generation due to their ability to utilize extensive embedded knowledge, yet they often require an enormous number of parameters to handle very long sequences, creating efficiency and scalability constraints. I questioned how to retain deep contextual reasoning while mitigating these limitations, prompting me to investigate State Space Models like Mamba that effectively manage long sequences with fewer parameters. How could such an architecture maintain extensive context without the overhead of massive parameter sets? I discovered that Mamba demonstrated robust performance on tasks with larger input sizes, whereas transformer-based approaches excelled at shorter contexts, suggesting a direct advantage for Mamba in long-sequence hypothesis generation. I conducted experiments comparing T5, GPT-4, and Mamba, noticing that GPT-4 consistently led in most evaluations, but Mamba excelled when inputs were extended, confirming the potential for tasks requiring greater context. Why did Mamba exhibit occasional instability on smaller inputs, and could the observed training loss behavior be related to vanishing or exploding gradients in a simplified neural design? Further data analysis showed no significant memorization artifacts, thereby validating the authenticity of novel hypothesis outputs. My turning point came when I realized that a refined, long-context architecture such as Mamba could unify scalable hypothesis generation with minimal parameter growth, providing a more sustainable path forward. In future experiments, I plan to focus on tuning hyperparameters for smaller input tasks, diversifying datasets beyond a single domain, and employing advanced state space variants to address instability concerns.
Use this template whenever you see a Bit statement, and produce the Spark, Flip, and Reasoning Chain as specified.


### Bit:
{}

### Spark:
{}


### Flip:
{}

### Reasoning Chain:

{}

"""

def formatting_prompts_func(examples):
    texts = []
    
    for bit, spark, flip, reason in zip(examples["bit"], examples["spark"], examples["flip"], examples["chain_of_reasoning"]):
        # Ensure that EOS token is added if necessary
        text = alpaca_prompt.format(bit, spark, flip, reason) + " <EOS>"
        texts.append(text)
    
    return {"text": texts}

# Load dataset
dataset = load_dataset("json", data_files="merged_reasoning.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

print(len(dataset["text"]))

# Load the model + tokenizer
model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache = True,
    device_map="auto", #{device_string}
)

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
max_seq_length = 1024
output_dir = "/scratch/ttc/sandeep/hypothesis"
per_device_train_batch_size = 4
optim = "adamw_hf"
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
warmup_ratio = 0.1
max_steps = len(dataset) * per_device_train_batch_size
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    optim=optim,
    max_steps=max_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=False, # Otherwise weird loss pattern (see https://github.com/artidoro/qlora/issues/84#issuecomment-1572408347, https://github.com/artidoro/qlora/issues/228, https://wandb.ai/answerdotai/fsdp_qlora/runs/snhj0eyh)
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=False,
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

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

exit(0)
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