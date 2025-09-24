import streamlit as st
import requests
import io
from PIL import Image
import os

# Настройки страницы
st.set_page_config(
    page_title="Food Classification",
    page_icon="🍕",
    layout="centered"
)

# Заголовок приложения
st.title("🍕 Food Classification")
st.markdown("Upload an image of food to classify it into one of the categories")

# Используем рабочий URL
API_URL = "http://host.docker.internal:8000"

def check_api_health():
    """Проверяет доступность API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def classify_image(image_bytes):
    """Отправляет изображение на API для классификации"""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def main():
    # Показываем какой URL используется
    st.sidebar.info(f"API URL: `{API_URL}`")
    
    # Проверка доступности API
    with st.spinner("Checking API connection..."):
        api_available = check_api_health()
    
    if not api_available:
        st.error(f"⚠️ Classification API is not available at {API_URL}")
        return

    st.success("✅ API is available and connected!")
    
    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Choose a food image", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image of food (max 10MB)"
    )
    
    if uploaded_file is not None:
        # Проверка размера файла
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size too large. Please upload an image smaller than 10MB.")
            return
        
        # Показ изображения
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Классификация при нажатии кнопки
        if st.button("Classify Food", type="primary"):
            with st.spinner("Analyzing image..."):
                # Чтение файла как bytes
                image_bytes = uploaded_file.getvalue()
                
                # Отправка на API
                result = classify_image(image_bytes)
                
                if result:
                    # Отображение результатов
                    st.success("Classification completed!")
                    
                    # Основной результат
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.metric(
                            label="Predicted Class", 
                            value=result["prediction"].upper(),
                            delta=f"{result['confidence']:.1%} confidence"
                        )
                    
                    # Детальные вероятности
                    with st.expander("View detailed probabilities"):
                        for class_name, prob in result["all_predictions"].items():
                            st.progress(prob, text=f"{class_name}: {prob:.1%}")

if __name__ == "__main__":
    main()