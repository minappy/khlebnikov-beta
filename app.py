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

# Загружаем характерные n-граммы и окказионализмы
with open('ngrams_stats.json', 'r', encoding='utf-8') as f:
    ngrams_stats = json.load(f)

okkazionalizms = ngrams_stats.get('okkazionalizms_list', [])
top_word_2grams = list(ngrams_stats.get('top_word_2grams', {}).keys())
top_char_3grams = list(ngrams_stats.get('top_char_3grams', {}).keys())

print(f"Loaded: {len(okkazionalizms)} okkazionalizms, {len(top_word_2grams)} word bigrams")

# Слова-триггеры Хлебникова (начинают генерацию в его стиле)
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


def inject_okkazionalizm(prompt):
    """Добавляет окказионализм в промпт если его нет"""
    if not prompt or not prompt.strip():
        return random.choice(KHLEBNIKOV_STARTERS)

    # Если в промпте уже есть окказионализм — оставляем
    for okkaz in okkazionalizms:
        if okkaz in prompt.lower():
            return prompt

    # Добавляем случайный окказионализм
    return random.choice(okkazionalizms) + " " + prompt if okkazionalizms else prompt


def generate(prompt, length, temp):
    # Если промпт пустой — используем характерное начало Хлебникова
    if not prompt or not prompt.strip():
        prompt = random.choice(KHLEBNIKOV_STARTERS)
    else:
        prompt = inject_okkazionalizm(prompt)

    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(length),
            temperature=float(temp),
            top_k=40,
            top_p=0.92,
            do_sample=True,
            num_return_sequences=3,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            early_stopping=False
        )

    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Разбиваем на строки как у Хлебникова
        lines = text.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        # Оставляем короткие строки (стиль Хлебникова)
        poetic_lines = [l for l in lines if len(l) < 80][:8]
        results.append('\n'.join(poetic_lines))

    return results[0], results[1], results[2]


print("\nStarting web interface...")
print("Open: http://127.0.0.1:7860")

with gr.Blocks(title="Khlebnikov Poetry Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎭 Генератор поэзии Хлебникова")
    gr.Markdown("*Оставьте поле пустым для случайной генерации в стиле будетлянина*")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Начало строки (или оставьте пустым)",
                placeholder="Оставьте пустым — модель сама начнёт в стиле Хлебникова",
                lines=2
            )
            length = gr.Slider(40, 200, value=80, step=10, label="Длина текста")
            temp = gr.Slider(0.8, 1.3, value=1.0, step=0.05, label="Температура (выше = экспериментальнее)")
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