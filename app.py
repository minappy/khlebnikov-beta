import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
import random

model_path = './model'
if not os.path.exists(model_path):
    print("Model not found. Run trainer.py first!")
    exit()

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model.eval()

with open('ngrams_stats.json', 'r', encoding='utf-8') as f:
    ngrams_stats = json.load(f)

okkazionalizms = ngrams_stats.get('okkazionalizms_list', [])

KHLEBNIKOV_STARTERS = [
    "Бобэоби пелись губы",
    "Крылышкуя золотописьмом",
    "О достоевскиймо бегущей тучи",
    "Смехачи смеялись",
    "Времир крылами",
    "Звеняш поюн",
    "Небобы пели",
    "Грёзога тихая",
    "Будетлянин шёл",
    "Могатырь ветряк",
]

def generate(prompt, length, temp):
    if not prompt or not prompt.strip():
        prompt = random.choice(KHLEBNIKOV_STARTERS)
    else:
        has_okkaz = any(o in prompt.lower() for o in okkazionalizms)
        if not has_okkaz and okkazionalizms:
            prompt = random.choice(okkazionalizms) + " " + prompt

    inputs = tokenizer(prompt, return_tensors='pt')
    max_tokens = int(length)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=float(temp),
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    lines = [l for l in lines if len(l) > 2]

    if max_tokens <= 50:
        lines = lines[:4]
    elif max_tokens <= 100:
        lines = lines[:6]
    else:
        lines = lines[:10]

    return '\n'.join(lines) if lines else text

print("\nStarting web interface...")
print("Open: http://127.0.0.1:7860")

with gr.Blocks(title="Khlebnikov Poetry Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎭 Генератор поэзии Хлебникова")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Начало строки",
                placeholder="Оставьте пустым для авто-генерации",
                lines=2
            )
            length = gr.Slider(40, 200, value=100, step=10, label="Длина")
            temp = gr.Slider(0.8, 1.3, value=1.0, step=0.05, label="Температура")
            btn = gr.Button("✨ Сгенерировать", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Стихотворение", lines=10, interactive=False)

    btn.click(fn=generate, inputs=[prompt, length, temp], outputs=output)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)