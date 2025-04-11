import numpy as np
def input_matrix():
    n = int(input("Nhập kích thước ma trận vuông n x n: "))
    matrix = []
    print("Nhập từng dòng của ma trận, cách nhau bởi dấu cách:")
    for i in range(n):
        row = list(map(float, input(f"Dòng {i+1}: ").split()))
        matrix.append(row)
    return np.array(matrix)
def gauss_jordan_inverse(matrix):

    """
    Tìm ma trận nghịch đảo bằng phương pháp Gauss-Jordan.
    
    :param matrix: Ma trận vuông (numpy array)
    :return: Ma trận nghịch đảo nếu tồn tại, ngược lại báo lỗi
    """
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]

    # Tạo ma trận mở rộng [A | I]
    augmented_matrix = np.hstack((matrix, np.eye(n)))

    # Áp dụng phép biến đổi hàng
    for i in range(n):
        # Tìm phần tử chính (pivot)
        pivot = augmented_matrix[i, i]
        if abs(pivot) < 1e-10:  # Kiểm tra nếu phần tử pivot quá nhỏ hoặc bằng 0
            raise ValueError("Ma trận không khả nghịch!")

        # Chia hàng i cho pivot để pivot trở thành 1
        augmented_matrix[i] = augmented_matrix[i] / pivot

        # Dùng hàng i để khử các phần tử khác trong cột
        for j in range(n):
            if i != j:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] -= factor * augmented_matrix[i]

    # Phần bên phải là ma trận nghịch đảo
    inverse_matrix = augmented_matrix[:, n:]
    return inverse_matrix
def gram_schmidt(V):
    """
    Thực hiện trực chuẩn hóa Gram-Schmidt trên tập hợp vector V.

    Args:
        V: Ma trận numpy (m x n), mỗi cột là một vector đầu vào.

    Returns:
        U: Ma trận chứa các vector trực chuẩn.
    """
    V = np.array(V, dtype=np.float64)
    m,n = V.shape  
    U = []
    for i in range(n):
        u_i = V[:, i]  # Lấy vector hiện tại

        for u_j in U:
            # Loại bỏ thành phần chiếu lên các vector trước đó
            proj = (np.dot(u_j, u_i)/np.dot(u_j,u_j)) * u_j
            u_i = u_i - proj

        
        norm_u_i = np.linalg.norm(u_i) # Chuẩn của vector
        if norm_u_i > 1e-10:  # Kiểm tra nếu vector không bị triệt tiêu hoàn toàn
            U.append(u_i/norm_u_i)  # Chuẩn hóa vector

    return np.array(U).T  # Chuyển dạng ma trận để mỗi cột là một vector trực chuẩn
def is_linear_mapping(A):
    """ Kiểm tra ảnh của ma trận có phải là ánh xạ tuyến tính không. """
    return True  # Mọi ánh xạ biểu diễn bởi ma trận đều là tuyến tính

def find_image(A):
    """ Tìm ảnh của ánh xạ tuyến tính (Im(f)) (tức là không gian cột của ma trận A). """
    U, S, Vt = np.linalg.svd(A)  # Phân rã SVD để tìm các vector độc lập
    rank = np.linalg.matrix_rank(A)  # Hạng của A
    image_basis = U[:,:rank]  # Cơ sở của không gian cột (Imf)
    return image_basis

def find_kernel(A):
    """ Tìm hạt nhân của ánh xạ tuyến tính (Ker(f)), tức là tập nghiệm của Ax = 0. """
    m,n = A.shape
    U, S, Vt = np.linalg.svd(A)  # Phân rã SVD để tìm không gian null
    null_space = Vt[len(S):].T  # Vector tương ứng với giá trị suy biến
    return null_space
def rank_of_vectors(dim,num_vectors):
    """
    Tính hạng của một hệ vector trong R^n
    :param dim: Chiều không gian của vector (số phần tử trong mỗi vector)
    :param num_vectors: Số lượng vector trong hệ
    :return: Hạng của hệ vector
    """
    vectors = []

    print(f"Nhập {num_vectors} vector trong không gian R^{dim}")
    for i in range(num_vectors):
        hung = list(map(float,input(f"\nNhập v{i+1}:").split()))
        if len(hung) != dim:
            print(f"Lỗi: khi nhập v{i+1} !")
            return
        vectors.append(hung)
    matrix = np.array(vectors)
    rank = np.linalg.matrix_rank(matrix)
    print(f"\nMa trận ta có:\n {matrix}")
    print(f"\nHạng của hệ vector là : {rank}")
    print(f"\nMa trận khi chuyển mỗi vector thành cột:\n {np.column_stack(matrix)}")
    print(f"\nHạng ma trận chuyển: {np.linalg.matrix_rank(np.column_stack(matrix))}")

















