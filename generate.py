from PIL import Image
import numpy as np
import os

SUPPORTED_FILE_TYPES = (
    '.png', 
    '.jpg', 
    '.jpeg', 
    '.bmp', 
    '.gif'
)

def main():
    image_folder = "."
    num_images = 5
    img_matrices = load_images(image_folder, num_images)

    avg_matrix = average_matrix(img_matrices)
    image = Image.fromarray(avg_matrix)
    image.save("output.png")


def load_images(image_folder, num_images):
    """
    Load in images from files in /images
    """
    image_matrices = []
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    image_files.sort() 
    
    for image_file in image_files:
        if image_file.lower().endswith(SUPPORTED_FILE_TYPES):
            image_matrices.append(np.array(Image.open(os.path.join(image_folder, image_file))))
    
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
    """
    Finds the average matrix from the matrix list using the average_color function
    """
    assert len(matrix_list) >= 1
    assert size_correct(matrix_list)

    rows = matrix_list[0].shape[0]
    cols = matrix_list[0].shape[1]
    result = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            colors_ij = [matrix[i][j] for matrix in matrix_list]
            result[i][j] = average_color(colors_ij)
    
    return result


def average_color(colors):
    """
    Takes in a list of RGB numpy vectors represented as [R, G, B] and returns an
    RGB vector corresponding to their average.
    """
    assert all((isinstance(color, np.ndarray) and color.ndim == 1) for color in colors)
    return np.mean(np.array(colors), axis=0)

if __name__ == "__main__":
    main()