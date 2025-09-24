import os
import cv2
import numpy as np

def load_and_preprocess_data_improved(data_path, img_size=(224, 224)):

    images = []
    labels = []
    
    # Получаем список папок-классов
    class_names = sorted([d for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))])
    
   

    
    total_loaded = 0
    total_errors = 0
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        

        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Загрузка изображения
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Ошибка загрузки: {img_path}")
                    total_errors += 1
                    continue
                
                # Проверка что изображение не пустое
                if img.size == 0:
                    print(f"Пустое изображение: {img_path}")
                    total_errors += 1
                    continue
                
                # Предобработка
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(class_idx)
                total_loaded += 1
                
            except Exception as e:
                print(f"Ошибка обработки {img_path}: {e}")
                total_errors += 1
    
    
    return np.array(images), np.array(labels), class_names



load_and_preprocess_data_improved('./data/train')