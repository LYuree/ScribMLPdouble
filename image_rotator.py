from PIL import Image, ImageOps
import os
import math

# Папка с исходными изображениями
source_folder = "numbers_dataset"

# Папка, куда будут сохранены сгенерированные изображения
destination_folder = "expanded_dataset2"

# Углы для поворота изображений
angles = list(range(-100, 95, 15))

# Создаем папку для расширенного датасета, если она не существует
current_directory = os.getcwd()
destination_path = os.path.join(current_directory, destination_folder)
os.makedirs(destination_path, exist_ok=True)

# Проходим по всем файлам в папке с исходными изображениями
for file_name in os.listdir(os.path.join(current_directory, source_folder)):
    # Проверяем, что файл является изображением
    if file_name.endswith(".png"):
        # Открываем исходное изображение
        image_path = os.path.join(current_directory, source_folder, file_name)
        image = Image.open(image_path)

        # Создаем папку для текущего изображения в расширенном датасете
        image_folder = os.path.join(destination_path, os.path.splitext(file_name)[0])
        os.makedirs(image_folder, exist_ok=True)

        # Проходим по всем углам и генерируем повернутые изображения
        for angle in angles:
            # Поворачиваем изображение на заданный угол и заполняем пустоты белыми пикселями
            rotated_image = image.rotate(angle/10, fillcolor='white', expand=True)

            width, height = rotated_image.size
            left = (width - 30) // 2
            top = (height - 30) // 2
            right = left + 30
            bottom = top + 30
            cropped_image = rotated_image.crop((left, top, right, bottom))

            # Генерируем имя файла для повернутого и обрезанного изображения
            rotated_file_name = f"{os.path.splitext(file_name)[0]}_{angle}.png"

            # Сохраняем повернутое и обрезанное изображение
            rotated_image_path = os.path.join(image_folder, rotated_file_name)
            cropped_image.save(rotated_image_path)

print("Дополнительные изображения успешно сгенерированы.")
