from PIL import Image
import numpy as np
import os

def main():
    image_folder = "."
    num_images = 5
    img_matrices = load_images(image_folder, num_images)

    avg_matrix = average_matrix(img_matrices)

    print(avg_matrix)
    # image = Image.fromarray(avg_matrix)
    # image.save("output.png")


def load_images(image_folder, num_images):
    """
    Load in images from files

    TODO: Make this better
    """
    image_matrices = []
    for i in range(1, num_images + 1):
        image_path = os.path.join(image_folder, f"{i}.png")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image_matrix = np.array(image)
            image_matrices.append(image_matrix)
        else:
            print(f"Image {i}.png not found in {image_folder}")
    return image_matrices


def size_correct(matrix_list):
    """
    Check if all matrices in matrix_list have the same size.
    """
    if len(matrix_list) == 0: return True 
    
    first_shape = matrix_list[0].shape
    for matrix in matrix_list[1:]:
        if matrix.shape != first_shape:
            return False
    return True


def average_matrix(matrix_list):
    assert(size_correct(matrix_list))
    for matrix in matrix_list:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

    

def average_color(colors):
    """
    Takes in a list of RGB numpy vectors represented as [R, G, B] and returns an
    RGB vector corresponding to their average.
    """
    assert all((isinstance(color, np.ndarray) and color.ndim == 1) for color in colors)
    return np.mean(np.array(colors), axis=0)

if __name__ == "__main__":
    main()