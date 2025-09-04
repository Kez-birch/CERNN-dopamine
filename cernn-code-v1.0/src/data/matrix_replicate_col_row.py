import numpy as np


def load_matrix(filename):
    return np.loadtxt(filename)


def replicate_row_col(matrix, times, position):
    # Validate the input
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError("Input must be a square numpy array")
    if position < 0 or position >= matrix.shape[0]:
        raise ValueError("Position must be within the range of the matrix dimensions")
    if times <= 0:
        raise ValueError("Times must be a positive integer")

    # Replicate the specified column
    replicated_column = np.tile(matrix[:, position : position + 1], times)
    new_matrix = np.hstack(
        (matrix[:, :position], replicated_column, matrix[:, position + 1 :])
    )

    # Replicate the specified row of the new matrix
    replicated_row = np.tile(new_matrix[position : position + 1, :], (times, 1))
    final_matrix = np.vstack(
        (new_matrix[:position, :], replicated_row, new_matrix[position + 1 :, :])
    )

    return final_matrix


def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)


def save_matrix(matrix, original_filename, x):
    new_filename = f"{original_filename.split('.txt')[0]}-{x}V1_units.txt"
    np.savetxt(new_filename, matrix, fmt="%0.4f")


# Example usage
filename = "C:/Users/mj23905/OneDrive - University of Bristol/Documents/GitHub/cortically-embedded-rnn/multitask/data/LeftParcelGeodesicDistmat.txt"
matrix = load_matrix(filename)
# matrix = np.array([[0, 2, 1, 3],
#                    [2, 0, 4, 4],
#                    [1, 4, 0, 0],
#                    [3, 4, 0, 0]])
times = int(input("Enter the number of times to replicate: "))
position = int(input("Enter the position to replicate: "))

resulted_matrix = replicate_row_col(
    matrix, times, position - 1
)  # Subtract 1 because of zero-based indexing
print(resulted_matrix)

save_matrix(resulted_matrix, filename, times)
# new matrix dimension = (180+times-1) X (180+times-1)
