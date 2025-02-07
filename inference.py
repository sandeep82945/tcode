import torch
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
def generate_hypothesis(bit_statement, max_length=512, temperature=0.7, top_p=0.9):
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

# Example inference
if __name__ == "__main__":
    bit_example = "Transformer-based models struggle with handling long sequences efficiently in scientific hypothesis generation."
    generated_hypothesis = generate_hypothesis(bit_example)
    print("\nGenerated Flip and its Reasoning Chain:")
    print(generated_hypothesis)
