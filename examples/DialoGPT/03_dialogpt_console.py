from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def chat_with_gpt():
    model.eval()
    print("Ask a question (or type 'exit' to stop):")
    input_text = "Hello, how are you?"
    while True:
        if input_text.lower() == 'exit':
            break

        # Encode and send to model
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Chatbot: {response}")
        input_text = input("You: ")

if __name__ == "__main__":
    # Load the trained model and tokenizer
    model_path = './build/models/dialogpt_model/'
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    chat_with_gpt()

