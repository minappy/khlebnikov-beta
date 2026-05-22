import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

texts = []
with open('corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        texts.append(json.loads(line)['text'])

print(f"Loaded {len(texts)} texts")

model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

encodings = tokenizer(texts, truncation=True, max_length=256, padding='max_length', return_tensors='pt')
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': encodings['input_ids'].clone()
})

split = dataset.train_test_split(test_size=0.1, seed=42)

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

print("Training...")
trainer.train()

model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
print("Done!")