import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os



class FoodCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(FoodCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 128x128 → 64x64
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 64x64 → 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 32x32 → 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Глобальный пулинг до 1x1
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ ДАННЫХ (с меньшим разрешением)
def load_data(data_dir, batch_size=16, img_size=128):  # Уменьшил разрешение!
    """
    Загружает данные из папок train и test
    """
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Папка {train_dir} не существует!")
    if not os.path.exists(test_dir):
        raise ValueError(f"Папка {test_dir} не существует!")
    
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    
    # Упрощенные трансформы для скорости
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Меньшее разрешение!
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Меньшее разрешение!
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка датасетов
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    # DataLoader'ы с меньшим batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)  # num_workers=0 для Mac
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0)  # num_workers=0 для Mac
    
    class_names = train_dataset.classes
    print(f"Найдено классов: {len(class_names)}")
    print(f"Классы: {class_names}")
    print(f"Тренировочных изображений: {len(train_dataset)}")
    print(f"Тестовых изображений: {len(test_dataset)}")
    
    return train_loader, test_loader, class_names

def train_improved(model, train_loader, test_loader, num_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    model = model.to(device)
    
    # Улучшенный optimizer с разными learning rate
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    # Планировщик с warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[1e-3, 1e-2], 
        epochs=num_epochs, 
        steps_per_epoch=len(train_loader)
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    best_accuracy = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            # Train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Statistics
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy
            }, 'best_model.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'Best Acc: {best_accuracy:.2f}%')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 60)
    
    return train_losses, val_accuracies, best_accuracy


if __name__ == "__main__":
    # Параметры для быстрой тренировки
    DATA_DIR = "./code/data"  # Твой путь
    BATCH_SIZE = 16     
    NUM_EPOCHS = 30     
    IMG_SIZE = 128       
    
    try:
        # Загрузка данных
        print("Загрузка данных...")
        train_loader, test_loader, class_names = load_data(
            DATA_DIR, BATCH_SIZE, IMG_SIZE
        )
        
        # Создание ПРОСТОЙ модели
        print("Создание модели...")
        model = FoodCNN(num_classes=len(class_names))
        
        print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
        
        # Быстрая тренировка (теперь получаем 3 значения вместо 1)
        print("Начало тренировки...")
        train_losses, val_accuracies, best_accuracy = train_improved(
            model, train_loader, test_loader, NUM_EPOCHS
        )
        
        
        print(f"Лучшая точность: {best_accuracy:.2f}%")
        
        # Сохранение модели
        torch.save(model.state_dict(), 'simple_food_model.pth')
        print("Модель сохранена!")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()