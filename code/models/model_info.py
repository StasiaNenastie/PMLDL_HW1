import json
import torch
import os

def save_model_info():
    """Сохраняет информацию о модели для API"""
    
    # Твои реальные классы
    class_names = ["pizza", "sushi", "hotdog", "hamburger"]
    
    model_info = {
        "class_names": class_names,
        "input_size": 128,
        "model_architecture": "FoodCNN",
        "num_classes": len(class_names),
        "display_names": {
            "pizza": "Пицца",
            "sushi": "Суши",
            "hotdog": "Хот-дог", 
            "hamburger": "Гамбургер"
        }
    }
    
    # Создаем папку models если её нет
    os.makedirs('models', exist_ok=True)
    
    # Сохраняем метаданные
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print("✅ Метаданные модели сохранены!")
    print(f"Классы: {class_names}")

if __name__ == "__main__":
    save_model_info()