import os
from PIL import Image
import csv
import math

def convert(images_folder, outf):
    o = open(outf, "w")
    images = []
    current_directory = os.getcwd()
    print(current_directory)
    i = 0 # for y
    for image_folder in os.listdir(os.path.join(current_directory, images_folder)):
        for file_name in os.listdir(os.path.join(images_folder, image_folder)):
            if file_name.endswith(".png"):
                image_path = os.path.join(images_folder, image_folder, file_name)
                image = Image.open(image_path)
                pixel_values = list(image.getdata())
                pixel_values = [str(abs(255 - pixel[0])) for pixel in pixel_values]
                o.write(f"{math.trunc(i/1000)}," + ",".join(str(pix) for pix in pixel_values)+"\n")
                i += 1

    # img_names = os.scandir(os.path.join(current_directory, img_dir))
    # csv_writer = csv.writer(csv_file)
    # with open(train_csv_path, mode='w', newline='') as csv_file:
    #     for i in range (0, n):
    #         image_path = os.path.join(img_dir, next(img_names))
    #         image = Image.open(image_path)
    #         pixel_values = list(image.getdata())
    #         pixel_values = [str(abs(255 - pixel[0])) for pixel in pixel_values]
    #         csv_writer.writerow(pixel_values)

    # for image in images:
    #     o.write(",".join(str(pix) for pix in image)+"\n")
    # o.close()

convert("cropped_dataset2", "my_images.csv")

# convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
#         "mnist_train.csv", 9000)
# convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
#         "mnist_test.csv", 1000)