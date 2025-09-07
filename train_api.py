# test_api.py
import requests
import argparse

# Адрес нашего запущенного API
API_URL = "http://127.0.0.1:5000/predict?language=rus"

def test_prediction(image_path: str):
    """
    Отправляет изображение в API и выводит результат.
    """
    try:
        # Открываем файл в двоичном режиме
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/png')} # Можно использовать image/jpeg
            
            print(f"Sending request for image: {image_path}")
            response = requests.post(API_URL, files=files)
            
            # Проверяем статус ответа
            response.raise_for_status() 
            
            # Выводим результат
            result = response.json()
            print("--- API Response ---")
            print(f"Predicted Class ID: {result.get('class_id')}")
            print(f"Full response: {result}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the API: {e}")
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test the Traffic Sign Recognition API.")
    # parser.add_argument("image_path", type=str, help="Path to the image file to classify.")
    # args = parser.parse_args()
    
    test_prediction("D:\\archive\\Test\\01305.png")
