import os
import sys
import numpy as np

loss = 1e-5  # 容忍偏差，例如 fp16 的绝对/相对误差阈值
minimum = 1e-10  # 防止除以零的小量

def load_interleaved_complex(filename: str) -> np.ndarray:
    """加载实虚交错存储的二进制文件为复数数组"""
    # 1. 以 float32 读取原始数据
    data = np.fromfile(filename, dtype=np.float32)
    # 2. 验证数据长度为偶数（实虚成对）
    if len(data) % 2 != 0:
        raise ValueError(f"文件 {filename} 数据长度非偶数，无法解析为复数")
    # 3. 构造复数数组（实部在偶数索引，虚部在奇数索引）
    return data[::2] + 1j * data[1::2]

def verify_complex_result(cal_result_bin: str, golden_bin: str) -> bool:
    # 加载复数数据
    cal_result = load_interleaved_complex("output/output.bin")
    golden = load_interleaved_complex("output/golden.bin")
    
    # 验证形状一致
    if cal_result.shape != golden.shape:
        print(f"[ERROR] 形状不匹配: cal_result {cal_result.shape} vs golden {golden.shape}")
        return False
    
    # 计算误差
    diff = np.abs(cal_result - golden)  # 复数差的模
    abs_golden = np.abs(golden)
    abs_cal = np.abs(cal_result)
    
    # 绝对误差检查: |cal - golden| <= loss
    abs_ok = (diff <= loss)
    # 相对误差检查: |cal - golden| / (|golden| + minimum) <= loss
    rel_ok = (diff / (np.maximum(abs_golden, abs_cal) + minimum)) <= loss
    
    # 综合检查（绝对误差或相对误差满足其一即可）
    overall_ok = np.logical_or(abs_ok, rel_ok)
    
    # 查找首个错误位置
    error_indices = np.where(~overall_ok)[0]
    if len(error_indices) > 0:
        first_error = error_indices[0]
        cal_val = cal_result[first_error]
        golden_val = golden[first_error]
        print(f"[ERROR] 第 {first_error} 个元素超出误差范围:")
        print(f"  Cal: {cal_val.real:.6f} + {cal_val.imag:.6f}j")
        print(f"  Golden: {golden_val.real:.6f} + {golden_val.imag:.6f}j")
        print(f"  Abs误差: {diff[first_error]:.2e} (阈值={loss:.1e})")
        return False
    
    print("测试通过")
    return True

if __name__ == '__main__':
    verify_complex_result(sys.argv[1],sys.argv[2])
