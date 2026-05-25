import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

texts = []
with open('corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        t = json.loads(line)['text']
        # Добавляем маркеры начала и конца стиха
        texts.append("<s>" + t + "</s>")

# Утраиваем данные
texts = texts * 5
print(f"Loaded {len(texts)} texts")

model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Добавляем специальные токены
special_tokens = {'additional_special_tokens': ['<s>', '</s>']}
tokenizer.add_special_tokens(special_tokens)

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

encodings = tokenizer(texts, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': encodings['input_ids'].clone()
})

args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=15,
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    warmup_steps=1000,
    logging_steps=10,
    save_strategy='no',
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Training...")
trainer.train()

model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
print("Done!")