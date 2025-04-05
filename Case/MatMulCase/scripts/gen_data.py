import torch
import torch.nn as nn
import numpy as np
import os

m = 128
n = 256
k = 64
case_data = {
    'case1': {
        'A_shape': [m, k],
        'data_type': np.float32,
        'B_shape': [k, n],
        'out_shape': [m, n]
    }
}

def case1():
    caseNmae='case1'
    tensor_input_real = np.random.uniform(1, 10,case_data[caseNmae]['A_shape']).astype(case_data[caseNmae]['data_type'])
    tensor_input_imag = np.random.uniform(1, 10,case_data[caseNmae]['A_shape']).astype(case_data[caseNmae]['data_type'])

    
    tensor_values_real = np.random.uniform(1, 10,case_data[caseNmae]['B_shape']).astype(case_data[caseNmae]['data_type'])
    tensor_values_imag = np.random.uniform(1, 10,case_data[caseNmae]['B_shape']).astype(case_data[caseNmae]['data_type'])
    
    complex_tensor_A = torch.complex(torch.from_numpy(tensor_input_real), torch.from_numpy(tensor_input_imag))
    complex_tensor_B = torch.complex(torch.from_numpy(tensor_values_real), torch.from_numpy(tensor_values_imag))
    
    tensor_bias_npu = None  
    res = torch.matmul(complex_tensor_A, complex_tensor_B)

    np_complex_x = complex_tensor_A.numpy().astype(np.complex64)
    np_complex_y = complex_tensor_B.numpy().astype(np.complex64)
    golden = res.numpy().astype(np.complex64)
    # 导出二进制文件
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    np_complex_x .tofile("./input/input_x.bin")
    np_complex_y.tofile("./input/input_y.bin")  # 替换原input_index.bin
    golden.tofile("./output/golden.bin")



if __name__ == "__main__":
    case1()