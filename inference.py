import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the fine-tuned model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Base model name
fine_tuned_model_dir = "/scratch/ttc/sandeep/hypothesis"  # Directory where trained model is saved

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto")

# Load PEFT adapter (LoRA fine-tuned model)
model = PeftModel.from_pretrained(model, fine_tuned_model_dir)
model = model.merge_and_unload()
model.eval()

# Function to generate response
def generate_hypothesis(bit_statement, max_length=1024, temperature=0.1, top_p=0.9):
    """Generates a hypothesis (Flip, and its Reasoning Chain) given a Bit statement."""
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
test_file = "test_set_reasoning.json"
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
output_file = "generated_outputs.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Generated outputs saved to {output_file}")
