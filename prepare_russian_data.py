import torch
import numpy as np
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import pickle

print("🚀 Загружаем русский датасет...")
try:
    # Загружаем датасет - только первые 5000 примеров для скорости
    ds = load_dataset("rustensai/russian-handwriting-ocr", split="train[:5000]", trust_remote_code=True)
    print(f"✅ Загружено {len(ds)} примеров")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    print("Пытаемся загрузить с другими параметрами...")
    ds = load_dataset("rustensai/russian-handwriting-ocr", trust_remote_code=True)
    if isinstance(ds, dict):
        ds = ds['train']
    ds = ds.select(range(min(5000, len(ds))))
    print(f"✅ Загружено {len(ds)} примеров")

# Исследуем структуру
print("\n📊 Структура датасета:")
sample = ds[0]
print(f"Ключи: {sample.keys()}")

# Определяем, какой ключ содержит изображение и текст
image_key = None
text_key = None

for key in sample.keys():
    val = sample[key]
    if hasattr(val, 'shape') and len(val.shape) > 1:  # Вероятно изображение
        image_key = key
        print(f"  {key}: shape={val.shape} (вероятно изображение)")
    elif isinstance(val, str):
        text_key = key
        print(f"  {key}: {val[:50]}... (вероятно текст)")
    else:
        print(f"  {key}: {type(val)}")

# Создаём выходную папку
output_dir = Path('.')
print(f"\n💾 Сохраняем данные в {output_dir}...")

# Собираем все тексты
sentences = []
strokes_list = []

print("\n⏳ Обработка данных...")
for i, example in enumerate(tqdm(ds, desc="Обработка примеров")):
    try:
        # Извлекаем текст
        if text_key and text_key in example:
            text = example[text_key]
            if isinstance(text, str) and len(text) > 0:
                sentences.append(text)
            else:
                continue
        
        # Извлекаем "stroke" данные - если есть
        # Это может быть изображение, которое мы преобразуем в strokes
        if image_key and image_key in example:
            img = example[image_key]
            
            # Преобразуем изображение в массив numpy если нужно
            if hasattr(img, 'convert'):  # PIL Image
                img = np.array(img)
            elif not isinstance(img, np.ndarray):
                img = np.array(img)
            
            # Создаём синтетические stroke данные из изображения
            # Используем градиент изображения как приближение stroke
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)  # Конвертируем в grayscale
            
            # Создаём простые coordinates из пикселей
            coords = []
            h, w = img.shape
            
            for y in range(0, h, max(1, h//50)):  # Сэмплируем каждые ~50 пикселей
                for x in range(0, w, max(1, w//50)):
                    if img[y, x] > 128:  # Если пиксель не белый
                        coords.append([float(x), float(y), 0.0])
            
            if len(coords) > 5:
                coords.append([float(w), float(h), 1.0])  # Конец stroke
                strokes_list.append(np.array(coords, dtype=np.float32))
            else:
                # Если изображение слишком светлое, пропускаем
                sentences.pop()
                continue
        
    except Exception as e:
        print(f"⚠️  Ошибка на примере {i}: {e}")
        if sentences and len(sentences) > len(strokes_list):
            sentences.pop()
        continue

# Синхронизируем длины
min_len = min(len(sentences), len(strokes_list))
sentences = sentences[:min_len]
strokes_list = strokes_list[:min_len]

print(f"\n✅ Обработано: {len(sentences)} примеров")

if len(sentences) == 0:
    print("❌ Нет данных! Проверьте формат датасета.")
    exit(1)

# Сохраняем sentences.txt
print("💾 Сохраняю sentences.txt...")
with open(output_dir / 'sentences.txt', 'w', encoding='utf-8') as f:
    for sent in sentences:
        f.write(sent.strip() + '\n')

# Сохраняем strokes.npy
print("💾 Сохраняю strokes.npy...")
strokes_array = np.array(strokes_list, dtype=object)
np.save(output_dir / 'strokes.npy', strokes_array, allow_pickle=True)

print(f"\n✅ Готово!")
print(f"   - sentences.txt: {len(sentences)} строк")
print(f"   - strokes.npy: {len(strokes_array)} примеров")
print(f"\n📝 Примеры текстов:")
for i in range(min(3, len(sentences))):
    print(f"   {i+1}. {sentences[i][:60]}")
