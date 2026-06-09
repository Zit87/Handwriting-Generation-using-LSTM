import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import HandwritingGenerator
from data import GetDataset
import matplotlib.pyplot as plt

# Параметры обучения
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 400
NUM_MIXTURES_ATTN = 10
NUM_MIXTURES_OUTPUT = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 Обучение на {DEVICE}")
print(f"Параметры:")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Device: {DEVICE}")

# Загружаем датасет
print("\n📊 Загружаю датасет...")
try:
    dataset = GetDataset('.', split='train')
    vocab_size = dataset.vocab_size
    print(f"✅ Датасет загружен! Размер словаря: {vocab_size}")
except Exception as e:
    print(f"❌ Ошибка загрузки датасета: {e}")
    exit(1)

# Создаём модель
print("\n🏗️  Создаю модель...")
model = HandwritingGenerator(
    vocab_size=vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=3,
    num_mixtures_attn=NUM_MIXTURES_ATTN,
    num_mixtures_output=NUM_MIXTURES_OUTPUT
).to(DEVICE)

print(f"✅ Модель создана!")
print(f"   Параметров: {sum(p.numel() for p in model.parameters()):,}")

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# DataLoader
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"\n📦 DataLoader готов!")
print(f"   Примеров: {len(dataset)}")
print(f"   Батчей: {len(train_loader)}")

# Обучение
print(f"\n🎯 Начинаю обучение на {EPOCHS} эпох...\n")

train_losses = []
best_loss = float('inf')

try:
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (char_seq, char_mask, strokes, strokes_mask) in enumerate(pbar):
            char_seq = char_seq.to(DEVICE)
            char_mask = char_mask.to(DEVICE)
            strokes = strokes.to(DEVICE)
            strokes_mask = strokes_mask.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Вычисляем losses
            stroke_loss, eos_loss, _, _, _ = model.losses(
                char_seq, char_mask, strokes, strokes_mask
            )
            
            total_loss = stroke_loss + eos_loss
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'stroke_loss': f'{stroke_loss.item():.4f}',
                'eos_loss': f'{eos_loss.item():.4f}'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f"\n✅ Эпоха {epoch+1} завершена - Loss: {avg_loss:.4f}")
        
        # Сохраняем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'model_russian.pth')
            print(f"💾 Модель сохранена! (loss: {avg_loss:.4f})")
        
        scheduler.step()
        
except KeyboardInterrupt:
    print("\n⚠️  Обучение прервано!")

except Exception as e:
    print(f"\n❌ Ошибка во время обучения: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Сохраняем финальную модель
    torch.save(model.state_dict(), 'model_russian_final.pth')
    print(f"\n✅ Обучение завершено!")
    print(f"   - Финальная модель: model_russian_final.pth")
    print(f"   - Лучшая модель: model_russian.pth (loss: {best_loss:.4f})")
    
    # Строим график потерь
    if train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
        print(f"   - График: training_loss.png")

print("\n🎉 Готово! Теперь можете тестировать с помощью generate_handwriting.py")
print(f"   Обновите путь к модели на: 'model_russian.pth'")
