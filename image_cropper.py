from PIL import Image, ImageOps
import os

# Папка с исходными изображениями
source_folder = "expanded_dataset2"

# Папка, куда будут сохранены вырезанные рамки
destination_folder = "cropped_dataset2"

# Создаем папку для вырезанных рамок, если она не существует
current_directory = os.getcwd()
destination_path = os.path.join(current_directory, destination_folder)
os.makedirs(destination_path, exist_ok=True)

# Проходим по всем папкам в папке с исходными изображениями
for image_folder in os.listdir(os.path.join(current_directory, source_folder)):
    # Создаем папку для текущего изображения в папке с вырезанными рамками
    cropped_image_folder = os.path.join(destination_path, image_folder)
    os.makedirs(cropped_image_folder, exist_ok=True)

    # Проходим по всем файлам в папке с текущим изображением
    for file_name in os.listdir(os.path.join(current_directory, source_folder, image_folder)):
        # Проверяем, что файл является изображением
        if file_name.endswith(".png"):
            # Открываем изображение
            image_path = os.path.join(current_directory, source_folder, image_folder, file_name)
            image = Image.open(image_path)

            # Проходим по всем координатам верхнего левого угла рамки
            for x in range(3):
                for y in range(3):
                    # Вырезаем рамку размером 28x28
                    left = x
                    top = y
                    right = left + 28
                    bottom = top + 28
                    cropped_image = image.crop((left, top, right, bottom))

                    # Сохраняем вырезанную рамку
                    cropped_file_name = f"{os.path.splitext(file_name)[0]}_{x}{y}.png"
                    cropped_image_path = os.path.join(cropped_image_folder, cropped_file_name)
                    cropped_image.save(cropped_image_path)

print("Рамки успешно вырезаны из изображений.")


