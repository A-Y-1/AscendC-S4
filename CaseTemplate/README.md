修改当前模板为任意算子的测试用例，共需要修改：
### 1
/CaseTemplate/inc/operator_desc.h中的struct OperatorDesc结构体，添加算子生成的build_out/autogen/aclnn_heaviside.h中，aclnnOPGetWorkspaceSize中除aclTensor和最后两个参数以外的其他参数。例如：

```c
__attribute__((visibility("default")))
aclnnStatus aclnnHeavisideGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *values,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

该算子输入输出均为Tensor，因此不需要添加任何数据结构到/CaseTemplate/inc/operator_desc.h中的struct OperatorDesc结构体当中。

### 2 

修改CaseTemplate/src/main.cpp中的CreateOpDesc，例如当前算子Heaviside需要一个379，379的浮点用例，修改为：

```c
OperatorDesc CreateOpDesc()
{
    // TODO 2
    // define operator
    int size = 379;
    std::vector<int64_t> shape_input {size, size};
    std::vector<int64_t> shape_values {size, size};
    std::vector<int64_t> shape_output {size,size};
    aclDataType dataTypeInput = ACL_FLOAT;  //输入输出类型
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    //设置属性值，该算子没有属性值，所以此处不需要
    // opDesc.dim = 1;
    //opDesc.reduce = "sum";
    //opDesc.include_self = true; 

    opDesc.AddInputTensorDesc(dataTypeInput, shape_input.size(), shape_input.data(), format);
    opDesc.AddInputTensorDesc(dataTypeInput, shape_values.size(), shape_values.data(), format);
    opDesc.AddOutputTensorDesc(dataTypeInput, shape_output.size(), shape_output.data(), format);
    return opDesc;

}
```

### 3
在CaseTemplate/src/op_runner.cpp顶部引入build_out/autogen/aclnn_算子名称.h头文件，不需要路径：

```c
#include "aclnn_heaviside.h"
```

### 4
在CaseTemplate/src/op_runner.cpp中修改算子的GetWorkSpace接口调用，后两个参数不需要修改，只需要修改前面的Tensor和其他值输入：

```c
//算子样例：ScatterReduce
auto ret = aclnnScatterReduceGetWorkspaceSize(inputTensor_[0], inputTensor_[1],inputTensor_[2],opDesc_->dim,opDesc_->reduce,opDesc_->include_self, outputTensor_[0], &workspaceSize, &handle);
//算子样例：Heaviside
auto ret = aclnnHeavisideGetWorkspaceSize(inputTensor_[0], inputTensor_[1], outputTensor_[0], &workspaceSize, &handle);
```

### 5 
在CaseTemplate/src/op_runner.cpp中修改算子接口调用，只需要替换名称：

```c
ret = aclnnHeaviside(workspace, workspaceSize, handle, stream);
```

至此算子已可以正常调用，接下来只需要修改gen_data.py和verify_result.py，修改golden结果生成为pytorch调用即可测试。