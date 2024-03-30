import os
import shutil

# Получаем текущую директорию
current_directory = os.getcwd()

# Создаем папку numbers_dataset, если её нет
dataset_folder = "numbers_dataset"
dataset_path = os.path.join(current_directory, dataset_folder)
os.makedirs(dataset_path, exist_ok=True)

# Перемещаем изображения из папки my_data в numbers_dataset
source_folder = ""
source_path = os.path.join(current_directory, source_folder)
for file_name in os.listdir(source_path):
    if file_name.endswith("bw.png"):
        source_file_path = os.path.join(source_path, file_name)
        destination_file_path = os.path.join(dataset_path, file_name)
        shutil.move(source_file_path, destination_file_path)

print("Изображения успешно перемещены в папку numbers_dataset.")


print(os.getcwd())
