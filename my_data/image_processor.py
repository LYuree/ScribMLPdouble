from PIL import Image
import os

# Получаем текущую директорию
current_directory = os.getcwd()
print(current_directory)
images_folder = "my_data"
images_path = os.path.join(current_directory, images_folder)

# Проходим по всем файлам в текущей директории
for file_name in os.listdir(images_path):
    # Проверяем, что файл имеет расширение .png
    # print(file_name)
    if file_name.endswith(".png"):
        # Открываем изображение
        image_path = os.path.join(images_path, file_name)
        image = Image.open(image_path)

        # Преобразуем изображение в черно-белое с альфа каналом
        bw_image = image.convert("LA")

        # Изменяем размер изображения на 28x28 пикселей
        resized_image = bw_image.resize((28, 28))

        # Сохраняем результат
        new_file_name = os.path.splitext(file_name)[0] + "_bw.png"
        print(new_file_name)
        resized_image.save(new_file_name)

print("Все изображения обработаны и сохранены.")
