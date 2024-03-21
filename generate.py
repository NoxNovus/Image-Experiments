from PIL import Image
import numpy as np
import os

IMAGE_FOLDER = "./images"

SUPPORTED_FILE_TYPES = (
    '.png', 
    '.jpg', 
    '.jpeg', 
    '.bmp', 
    '.gif'
)

WIDTH = 1920
HEIGHT = 1080

def main():
    resize()
    raw_matrices, filenames = load_images()
    img_matrices = size_correct(raw_matrices, filenames)
    avg_matrix = vector_average_matrix(img_matrices)
    image = Image.fromarray(avg_matrix)
    image.save("output.png")


def resize():
    """
    Check if the user would like to run resize operations on their images.
    """
    def resize_op():
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)

        files = os.listdir(IMAGE_FOLDER)

        for file_name in files:
            input_path = os.path.join(IMAGE_FOLDER, file_name)
            with Image.open(input_path) as img:
                resized_img = img.resize((1920, 1080), Image.Resampling.NEAREST)
                output_path = os.path.join(IMAGE_FOLDER, file_name)
                resized_img.save(output_path)

    user_input = input("Do you want to run the resize operation? (y/n): ").strip().lower()
    if user_input == "y":
        print("Resizing...")
        resize_op()
        print("Done.")
    elif user_input == "n":
        print("No resize operation will be performed.")
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")


def load_images():
    """
    Load in images from files in /images
    """    
    image_matrices = []
    index_to_filename = {}

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]
    image_files.sort() 
    
    for index, image_file in enumerate(image_files):
        if image_file.lower().endswith(SUPPORTED_FILE_TYPES):
            image_matrices.append(np.array(Image.open(os.path.join(IMAGE_FOLDER, image_file))))
            index_to_filename[index] = image_file
    
    return image_matrices, index_to_filename


def size_correct(matrix_list, filenames):
    """
    Check if all matrices in matrix_list have the same size.
    Removes any matrices that are not the same size as the first matrix.
    """
    result_list = []
    
    first_shape = matrix_list[0].shape
    for i, matrix in enumerate(matrix_list):
        if matrix.shape != first_shape:
            print(f"Image {filenames[i]} was misshaped, skipping")
        else:
            result_list.append(matrix)

    return result_list


def vector_average_matrix(matrix_list):
    assert len(matrix_list) >= 1
    return np.mean(np.stack(matrix_list, axis=-1), axis=-1).astype(np.uint8)


def average_color(colors):
    """
    Takes in a list of RGB numpy vectors represented as [R, G, B] and returns an
    RGB vector corresponding to their average.
    """
    assert all((isinstance(color, np.ndarray) and color.ndim == 1) for color in colors)
    return np.mean(np.array(colors), axis=0)

if __name__ == "__main__":
    main()