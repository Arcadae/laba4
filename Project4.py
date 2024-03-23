#ВАРИАНТ 20. Формируется матрица F следующим образом: 
#скопировать в нее А и  если в Е количество чисел, больших К в четных столбцах , чем произведение чисел в нечетных строках , то поменять местами С и Е симметрично, иначе С и В поменять местами несимметрично.
#При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, то вычисляется выражение:A*AT–K * F-1
#иначе вычисляется выражение (A-1+G-FТ)*K, где G-нижняя треугольная матрица, полученная из А .Выводятся по мере формирования А, F и все матричные операции последовательно.


import numpy as np
import matplotlib.pyplot as plt

def generate_matrix(n):
    return np.random.randint(-10, 10, size=(n, n))

def split_matrix(matrix):
    n = len(matrix)
    half = n // 2
    return matrix[:half, :half], matrix[:half, half:], matrix[half:, :half], matrix[half:, half:]

def count_even_cols(matrix,K):
    return np.sum(matrix[:, ::2] > K)

def count_odd_rows_product(matrix):
    return np.prod(matrix[1::2, :])

def swap_matrices_symmetrically(matrix1, matrix2):
    n = len(matrix1)
    for i in range(n):
        for j in range(i + 1, n):
            if i + j != n - 1:
                matrix1[i, j], matrix2[n - j - 1, n - i - 1] = matrix2[n - j - 1, n - i - 1], matrix1[i, j]

def swap_matrices(matrix1, matrix2):
    matrix1[:], matrix2[:] = matrix2[:], matrix1[:]

def main():
    try:
        K = int(input("Введите число K: "))
        N = int(input("Введите число N: "))
    except ValueError:
        print('Матрица не может быть размером из нецелого числа')
        return

    A = generate_matrix(N)
    print("Матрица A:")
    print(A)

    B, C, D, E = split_matrix(A)

    F = np.vstack((np.hstack((B, C)), np.hstack((D, E))))
    print("Матрица F:")
    print(F)

    if count_even_cols(F,K) > count_odd_rows_product(F):
        swap_matrices_symmetrically(C, E)
    else:
        swap_matrices(C, B)

    print("Матрица C после обмена:")
    print(C)
    print("Матрица E после обмена:")
    print(E)

    det_A = np.linalg.det(A)
    sum_diag_F = np.sum(np.diag(F))

    G = np.tril(A, -1)
    print("Матрица G:")
    print(G)

    if det_A > sum_diag_F:
        result = A @ A.T - K * np.linalg.inv(F)
    else:
        result = (np.linalg.inv(A) + G - F.T) * K

    print("Результирующая матрица:")
    print(result)

    plt.matshow(A)
    plt.title('Матрица A')
    plt.colorbar()
    plt.show()

    plt.matshow(F)
    plt.title('Матрица F')
    plt.colorbar()
    plt.show()

    plt.matshow(result)
    plt.title('Результат выражения')
    plt.colorbar()
    plt.show()
    
if __name__=="__main__":
    main()
