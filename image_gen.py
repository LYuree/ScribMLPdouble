from PIL import Image, ImageOps
import os

# Папка с данными
data_folder = "numbers_dataset"

# Получаем текущую директорию
current_directory = os.getcwd()

# Полный путь к папке с данными
data_path = os.path.join(current_directory, data_folder)

# Проходим по всем файлам в папке с данными
for file_name in os.listdir(data_path):
    # Проверяем, что файл является изображением и имеет правильное расширение
    if file_name.endswith("bw.png"):
        # Открываем изображение
        image_path = os.path.join(data_path, file_name)
        image = Image.open(image_path)

        # Проходим по каждому пикселю изображения
        width, height = image.size
        for x in range(width):
            for y in range(height):
                # Получаем значение яркости пикселя
                brightness = image.getpixel((x, y))[0]

                # Если яркость пикселя меньше 35, делаем пиксель полностью белым
                if brightness >= 180:
                    image.putpixel((x, y), (255, 255))

        # Добавляем белую рамку
        bordered_image = ImageOps.expand(image, border=1, fill='white')

        # Сохраняем измененное изображение
        bordered_image.save(image_path)


print("Обработка изображений завершена.")
