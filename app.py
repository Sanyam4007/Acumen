import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the fine-tuned model and tokenizer from the folder
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# Function to generate a response from the model
def generate_response(prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit app layout
st.title("AI Chatbot")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input)
    st.write("Bot:", response)
