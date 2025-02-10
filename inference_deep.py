import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


from unsloth import FastLanguageModel, is_bfloat16_supported

max_seq_length = 2048
dtype = None
load_in_4bit = True


fine_tuned_model_dir = "/scratch/ttc/sandeep/hypothesis_deep"

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    fine_tuned_model_dir, # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = load_in_4bit,
)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)

model.eval()

# Function to generate response
def generate_hypothesis(bit_statement, max_length=1024, temperature=0.1, top_p=0.9):
    """Generates only hypothesis (Flip, and its Reasoning Chain) given a Bit statement."""
    input_text = f"""
    ### Bit:
    {bit_statement}
    Generated Flip and its Reasoning Chain is:
    """
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Load test dataset
test_file = "test_final.json"
with open(test_file, "r") as f:
    test_data = json.load(f)

# Process test set and generate outputs
results = []
for item in test_data:
    bit_statement = item.get("bit", "")
    generated_output = generate_hypothesis(bit_statement)
    results.append({
        "title": item.get("title", ""),
        "bit": bit_statement,
        "generated_flip_reasoning": generated_output
    })

# Save results to JSON
output_file = "generated_outputs_deep.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Generated outputs saved to {output_file}")
