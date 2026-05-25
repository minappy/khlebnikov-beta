import json
import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Загрузка текстов
texts = []
with open('corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        texts.append(json.loads(line)['text'])

print(f"Loaded {len(texts)} texts")

# Загрузка статистики n-грамм (из corpus_builder)
try:
    with open('ngrams_stats.json', 'r', encoding='utf-8') as f:
        ngrams_stats = json.load(f)
    print(f"Loaded n-gram stats:")
    print(f"  Unique char 2-grams: {ngrams_stats['unique_char_2grams']}")
    print(f"  Unique char 3-grams: {ngrams_stats['unique_char_3grams']}")
    print(f"  Unique word 2-grams: {ngrams_stats['unique_word_2grams']}")
    print(f"  Unique word 3-grams: {ngrams_stats['unique_word_3grams']}")
    print(f"  Okkazionalizms: {ngrams_stats['okkazionalizms_found']}")
except:
    ngrams_stats = None
    print("No ngrams_stats.json found")

# Загрузка модели
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
print(f"\nLoading model: {model_name}")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Токенизация
print("Tokenizing...")
encodings = tokenizer(texts, truncation=True, max_length=256, padding='max_length', return_tensors='pt')
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': encodings['input_ids'].clone()
})

split = dataset.train_test_split(test_size=0.1, seed=42)
print(f"Train: {len(split['train'])}, Val: {len(split['test'])}")

# Вычисление perplexity до обучения
print("\nPerplexity BEFORE training:")
model.eval()
total_loss = 0
total_tokens = 0
with torch.no_grad():
    for text in texts[:10]:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        outputs = model(**inputs, labels=inputs['input_ids'])
        total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
        total_tokens += inputs['input_ids'].size(1)
ppl_before = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
print(f"  Perplexity: {ppl_before:.2f}")

# Обучение
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    warmup_steps=200,
    logging_steps=10,
    save_strategy='no',
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=split['train'],
    eval_dataset=split['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("\nTraining...")
trainer.train()

# Perplexity после обучения
print("\nPerplexity AFTER training:")
model.eval()
total_loss = 0
total_tokens = 0
with torch.no_grad():
    for text in texts[:10]:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        outputs = model(**inputs, labels=inputs['input_ids'])
        total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
        total_tokens += inputs['input_ids'].size(1)
ppl_after = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
print(f"  Perplexity: {ppl_after:.2f}")

# Сохранение модели
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

# Сохранение метрик
metrics = {
    'model': model_name,
    'texts_count': len(texts),
    'epochs': 3,
    'batch_size': 2,
    'learning_rate': 1e-4,
    'perplexity_before': round(ppl_before, 2),
    'perplexity_after': round(ppl_after, 2),
    'improvement': round(ppl_before - ppl_after, 2),
    'ngrams_stats': ngrams_stats
}

with open('metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО:")
print(f"  Perplexity до: {ppl_before:.2f}")
print(f"  Perplexity после: {ppl_after:.2f}")
print(f"  Улучшение: {ppl_before - ppl_after:.2f}")
print(f"{'='*50}")
print(f"Модель сохранена в ./model")
print(f"Метрики сохранены в metrics.json")