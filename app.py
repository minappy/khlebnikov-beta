import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

model_path = './model'
if not os.path.exists(model_path):
    print("Model not found. Run trainer.py first!")
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
            num_return_sequences=3,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )

    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        lines = text.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        results.append('\n'.join(lines[:8]))

    return results[0], results[1], results[2]

print("\nStarting web interface...")
print("Open: http://127.0.0.1:7860")

with gr.Blocks(title="Khlebnikov Poetry Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎭 Генератор поэзии Хлебникова")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Начало строки",
                placeholder="Бобэоби пелись губы...",
                lines=2
            )
            length = gr.Slider(30, 150, value=60, step=10, label="Длина текста")
            temp = gr.Slider(0.7, 1.2, value=0.9, step=0.05, label="Температура")
            btn = gr.Button("✨ Сгенерировать", variant="primary")

        with gr.Column(scale=2):
            out1 = gr.Textbox(label="Вариант 1", lines=6, interactive=False)
            out2 = gr.Textbox(label="Вариант 2", lines=6, interactive=False)
            out3 = gr.Textbox(label="Вариант 3", lines=6, interactive=False)

    btn.click(
        fn=generate,
        inputs=[prompt, length, temp],
        outputs=[out1, out2, out3]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)