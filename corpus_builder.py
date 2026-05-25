import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from collections import Counter
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize

# Скачиваем данные NLTK
nltk.download('punkt', quiet=True)

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

# Счётчики для всех n-грамм
all_char_2grams = Counter()
all_char_3grams = Counter()
all_word_2grams = Counter()
all_word_3grams = Counter()
okkazionalizms_found = set()

# Словарь окказионализмов
OKKAZIONALIZMS = {
    'бобэоби', 'вээоми', 'пиээо', 'лилээй', 'гзигзи', 'гзэгзэ',
    'крылышкуя', 'золотописьмом', 'лебедиво',
    'смехач', 'смеянство', 'смеяльно', 'смешики', 'смеюнчики',
    'времир', 'времыши', 'могатырь', 'будетлянин',
    'звучаль', 'звеняш', 'поюн', 'песнезвон', 'небобы',
    'грёзога', 'славка', 'любва', 'снежень', 'ручьёвина',
    'травник', 'цветун', 'морян', 'ветряк', 'солнцепах',
    'лунничать', 'звёздопадь', 'творянин', 'дружево',
}


def compute_char_ngrams(text, n):
    """Буквенные n-граммы"""
    clean = re.sub(r'[^а-яё]', '', text.lower())
    return Counter([clean[i:i + n] for i in range(len(clean) - n + 1)]) if len(clean) >= n else Counter()


def compute_word_ngrams(words, n):
    """Словесные n-граммы"""
    if len(words) < n:
        return Counter()
    return Counter(list(ngrams(words, n)))


def find_okkazionalizms(words):
    """Поиск окказионализмов в тексте"""
    found = set()
    for word in words:
        if word.lower() in OKKAZIONALIZMS:
            found.add(word.lower())
    return found


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

        # Очистка
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not re.match(r'^(предыдущий|следующий|содержание|вверх|назад|В\.|©|http)', line, re.IGNORECASE):
                if len(line) > 3:
                    lines.append(line)

        clean_text = '\n'.join(lines[:15])

        if len(clean_text) > 100:
            texts.append(clean_text)

            # === РАЗМЕТКА N-ГРАММ ===

            # Буквенные n-граммы
            char_2 = compute_char_ngrams(clean_text, 2)
            char_3 = compute_char_ngrams(clean_text, 3)
            all_char_2grams.update(char_2)
            all_char_3grams.update(char_3)

            # Токенизация слов
            words = word_tokenize(clean_text.lower())
            words = [w for w in words if w.isalpha()]

            # Словесные n-граммы
            word_2 = compute_word_ngrams(words, 2)
            word_3 = compute_word_ngrams(words, 3)
            all_word_2grams.update(word_2)
            all_word_3grams.update(word_3)

            # Окказионализмы
            okkaz = find_okkazionalizms(words)
            okkazionalizms_found.update(okkaz)

            print(
                f"OK {i + 1}/100 | chars: {len(clean_text)} | char_2: {len(char_2)} | char_3: {len(char_3)} | word_2: {len(word_2)} | word_3: {len(word_3)} | okkaz: {len(okkaz)}")

    except Exception as e:
        print(f"Skip {i + 1}: {e}")

# Сохраняем тексты
with open('corpus.jsonl', 'w', encoding='utf-8') as f:
    for t in texts:
        f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')

# Сохраняем статистику n-грамм
stats = {
    'total_poems': len(texts),
    'total_characters': sum(len(t) for t in texts),
    'unique_char_2grams': len(all_char_2grams),
    'unique_char_3grams': len(all_char_3grams),
    'unique_word_2grams': len(all_word_2grams),
    'unique_word_3grams': len(all_word_3grams),
    'okkazionalizms_found': len(okkazionalizms_found),
    'okkazionalizms_list': sorted(list(okkazionalizms_found)),
    'top_char_2grams': dict(all_char_2grams.most_common(50)),
    'top_char_3grams': dict(all_char_3grams.most_common(50)),
    'top_word_2grams': {f"{k[0]} {k[1]}": v for k, v in all_word_2grams.most_common(30)},
    'top_word_3grams': {f"{k[0]} {k[1]} {k[2]}": v for k, v in all_word_3grams.most_common(20)},
}

with open('ngrams_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 50}")
print(f"КОРПУС СОБРАН:")
print(f"  Стихов: {stats['total_poems']}")
print(f"  Буквенных 2-грамм: {stats['unique_char_2grams']}")
print(f"  Буквенных 3-грамм: {stats['unique_char_3grams']}")
print(f"  Словесных биграмм: {stats['unique_word_2grams']}")
print(f"  Словесных триграмм: {stats['unique_word_3grams']}")
print(f"  Окказионализмов: {stats['okkazionalizms_found']}")
print(f"  Окказионализмы: {', '.join(stats['okkazionalizms_list'])}")
print(f"{'=' * 50}")
print(f"Статистика сохранена в ngrams_stats.json")