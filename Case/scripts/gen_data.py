import torch
import torch.nn as nn
import numpy as np
import os


def case1():
    # 配置参数
    src_type = np.float32
    shape = (379, 379)  # 指定二维形状
    
    # 生成输入数据 (包含零值以触发Heaviside的values逻辑)
    np.random.seed(0)
    input_x = np.random.uniform(-1, 1, shape).astype(src_type)  # 包含负数、零、正数
    input_values = np.random.uniform(0, 1, shape).astype(src_type)  # Heaviside的替换值
    
    # 手动添加更多零值以增强测试覆盖
    input_x[input_x > -0.3] = 0  # 约30%的元素置零
    
    # 转换为PyTorch Tensor
    input_x_cpu = torch.from_numpy(input_x)
    input_values_cpu = torch.from_numpy(input_values)
    
    # 调用Heaviside算子
    res = torch.heaviside(input_x_cpu, input_values_cpu)
    
    # 生成Golden结果
    golden = res.numpy().astype(src_type)
    
    # 导出二进制文件
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_values.tofile("./input/input_values.bin")  # 替换原input_index.bin
    golden.tofile("./output/golden.bin")



if __name__ == "__main__":
    case1()