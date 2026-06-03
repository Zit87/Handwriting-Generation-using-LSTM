from datasets import load_dataset
import numpy as np
from pathlib import Path
import json

# Загружаем датасет
print("Загружаем датасет...")
ds = load_dataset("rustensai/russian-handwriting-ocr")

print(f"Структура датасета: {ds}")
print(f"\nПример данных:")
print(ds['train'][0])

# Создаём папку для сохранения
output_dir = Path('./russian_data')
output_dir.mkdir(exist_ok=True)

# Исследуем структуру данных
print("\n\nИсследование структуры...")
sample = ds['train'][0]
print(f"Ключи в sample: {sample.keys()}")

for key in sample.keys():
    print(f"{key}: {type(sample[key])}")
