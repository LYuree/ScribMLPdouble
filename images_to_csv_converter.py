import os
import csv
from PIL import Image

# Папка с изображениями
images_folder = "cropped_dataset2"

# Путь к CSV файлу
csv_file_path = "images_data2.csv"

# Открываем CSV файл для записи
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Записываем заголовки столбцов
    # csv_writer.writerow(['label'] + [f'pixel_{i}' for i in range(784)])

    # Проходим по всем изображениям в папке
    for image_folder in os.listdir(images_folder):
        # Получаем метку (цифру) из названия папки
        label = image_folder.split('_')[0]

        # Проходим по всем файлам в папке с изображением
        for file_name in os.listdir(os.path.join(images_folder, image_folder)):
            # Проверяем, что файл является изображением
            if file_name.endswith(".png"):
                # Открываем изображение
                image_path = os.path.join(images_folder, image_folder, file_name)
                image = Image.open(image_path)

                # Преобразуем изображение в массив пикселей
                # image.getpixel((x, y))[0]
                pixel_values = list(image.getdata())

                pixel_values = [str(abs(255 - pixel[0])) for pixel in pixel_values]

                # Записываем метку и значения пикселей в CSV файл
                csv_writer.writerow([label] + pixel_values)

print("Данные успешно записаны в файл CSV.")
