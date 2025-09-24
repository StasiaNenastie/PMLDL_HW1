from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import json
import os
from typing import Dict, List

app = FastAPI(title="Food Classification API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Определяем архитектуру модели (такая же как в тренировочном коде)
class FoodCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(FoodCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
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

# Глобальные переменные для модели и классов
model = None
class_names = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Загружает модель и метаданные"""
    global model, class_names
    
    try:
        # Загружаем метаданные
        with open('models/models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        class_names = model_info['class_names']
        num_classes = len(class_names)
        
        # Создаем модель
        model = FoodCNN(num_classes=num_classes)
        
        # Загружаем checkpoint
        checkpoint = torch.load('models/best_model.pth', map_location=device)
        
        # Проверяем структуру загружаемого файла
        if 'model_state_dict' in checkpoint:
            # Если это полный checkpoint, берем только веса модели
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Если это прямые веса
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        
        print(f"✅ Модель загружена. Классы: {class_names}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False

# Трансформы для изображений (такие же как при обучении)
def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_bytes):
    """Преобразует изображение в тензор"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = get_transform()
    return transform(image).unsqueeze(0)  # Добавляем batch dimension

@app.on_event("startup")
async def startup_event():
    """Загружаем модель при старте приложения"""
    if not load_model():
        raise RuntimeError("Не удалось загрузить модель")

@app.get("/")
async def root():
    return {
        "message": "Food Classification API", 
        "classes": class_names,
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Предсказание класса еды по изображению"""
    try:
        # Проверяем формат файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Файл должен быть изображением")
        
        # Читаем и проверяем изображение
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(400, "Пустой файл")
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(400, "Файл слишком большой")
        
        # Препроцессинг
        input_tensor = preprocess_image(image_bytes)
        input_tensor = input_tensor.to(device)
        
        # Предсказание
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Формируем ответ
        result = {
            "prediction": class_names[predicted_class],
            "confidence": round(confidence, 4),
            "all_predictions": {
                class_name: round(probabilities[i].item(), 4) 
                for i, class_name in enumerate(class_names)
            }
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки изображения: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Возвращает список доступных классов"""
    return {"classes": class_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)