import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re

url = "https://hlebnikov.ru/wp-content/uploads/tvorenia/mat/contents.htm#prosa"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
response.encoding = 'windows-1251'
soup = BeautifulSoup(response.text, 'html.parser')

poems = []
for link in soup.find_all('a', href=True):
    href = link['href']
    if href.endswith('.htm') and 'contents' not in href:
        poems.append(f"https://hlebnikov.ru/wp-content/uploads/tvorenia/mat/{href}")

print(f"Found {len(poems)} poems")

os.makedirs('corpus', exist_ok=True)
texts = []

for i, url in enumerate(poems[:100]):
    try:
        time.sleep(0.3)
        resp = requests.get(url, headers=headers)
        resp.encoding = 'windows-1251'
        s = BeautifulSoup(resp.text, 'html.parser')

        pre = s.find('pre')
        if pre:
            text = pre.get_text()
        else:
            text = s.body.get_text() if s.body else ""

        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not re.match(r'^(предыдущий|следующий|содержание|вверх|назад|В\.|©|http)', line, re.IGNORECASE):
                if len(line) > 3:
                    lines.append(line)

        text = '\n'.join(lines[:15])

        if len(text) > 100:
            texts.append(text)
            print(f"OK {i + 1}/100: {len(text)} chars - {lines[0][:50] if lines else ''}")
    except Exception as e:
        print(f"Skip {i + 1}: {e}")

with open('corpus.jsonl', 'w', encoding='utf-8') as f:
    for t in texts:
        f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')

print(f"\nSaved {len(texts)} poems")