from datasets import load_dataset
import numpy as np
from pathlib import Path

# Загружаем ТОЛЬКО небольшую часть датасета
print("Загружаем небольшую часть датасета...")
ds = load_dataset("rustensai/russian-handwriting-ocr", split="train[:1000]")  # Только первые 1000 примеров

print(f"Загружено примеров: {len(ds)}")
print(f"\nПример данных:")
print(ds[0])

print("\n\nИсследование структуры...")
sample = ds[0]
print(f"Ключи в sample: {sample.keys()}")

for key in sample.keys():
    val = sample[key]
    if hasattr(val, 'shape'):
        print(f"{key}: {type(val)} shape={val.shape}")
    else:
        print(f"{key}: {type(val)}")
