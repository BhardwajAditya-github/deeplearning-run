# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

model = model.half()

# Example input text
input_text = "Hello, I need to transfer money urgently, can you help me?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate model output
output = model.generate(**inputs, max_length=150)

# Decode the output (convert from tokens to human-readable text)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text: ", generated_text)