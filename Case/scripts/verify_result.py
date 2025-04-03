import os
import sys
import numpy as np

loss = 1e-5 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 10e-10
resType = np.float32

def verify_result(cal_result, golden):
    
    cal_result = np.fromfile(cal_result, dtype=resType) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=resType) # 从bin文件读取预期运算结果
    result = np.abs(cal_result - golden) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(cal_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss) # 计算相对误差
    
    # tol 为 1 说明符合误差范围
    
    # 检查错误位置
    for i in range(len(result_atol)):
        if not result_atol[i] or not result_rtol[i]:
            print(f"[ERROR] i={i}, cal={cal_result} , golden={golden}")
            return False
        
    # case 误差检查方法
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > cal_result.size * loss and np.sum(result_atol == False) > cal_result.size * loss: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
