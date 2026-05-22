import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

model_path = './model'
if not os.path.exists(model_path):
    print("Model not found. Run trainer.py first!")
    print("Looking for model in:", os.path.abspath(model_path))
    exit()

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model.eval()
print("Model loaded!")


def generate(prompt, length, temp):
    if not prompt or not prompt.strip():
        prompt = " "

    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(length),
            temperature=float(temp),
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if l.strip()]
    return '\n'.join(lines[:8])


print("Starting web interface...")
print("Open this URL in your browser: http://127.0.0.1:7860")

app = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Начало строки", placeholder="Бобэоби пелись губы", lines=2),
        gr.Slider(30, 150, value=60, label="Длина"),
        gr.Slider(0.7, 1.2, value=0.9, label="Температура")
    ],
    outputs=gr.Textbox(label="Результат", lines=10),
    title="Khlebnikov Poetry Generator"
)

app.launch(
    server_name="127.0.0.1",  # Явно указываем локальный адрес
    server_port=7860,
    share=False,
    quiet=False
)