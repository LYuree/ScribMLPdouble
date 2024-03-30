import os

# Получаем текущую директорию
current_directory = os.getcwd()

# Получаем список файлов в текущей директории
files = [file for file in os.listdir(current_directory) if file.lower().endswith('.png')]

# Сортируем список файлов по времени модификации
files.sort(key=lambda x: os.path.getmtime(os.path.join(current_directory, x)))

# Счетчик для отслеживания текущей группы файлов
group_counter = 1

# Счетчик для отслеживания количества файлов в текущей группе
file_counter = 0

# Проходим по всем файлам в отсортированном списке
for file_name in files:
    # Переименовываем файл, добавляя префикс в соответствии с текущей группой
    new_file_name = f"{group_counter}_{file_name}"
    os.rename(os.path.join(current_directory, file_name), os.path.join(current_directory, new_file_name))

    # Увеличиваем счетчик файлов в текущей группе
    file_counter += 1

    # Если достигнуто 10 файлов в текущей группе, переходим к следующей группе
    if file_counter == 10:
        group_counter += 1
        file_counter = 0

print("Файлы PNG переименованы в соответствии с заданным порядком.")
