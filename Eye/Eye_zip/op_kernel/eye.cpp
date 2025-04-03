#include "kernel_operator.h"

using namespace AscendC;
template<typename T>
class kernelEye{
public:
    __aicore__ inline kernelEye(){}
    __aicore__ inline void Init(GM_ADDR y, int numRows, int numCols, int blockPerCore, int nBigCore){
        int coreIdx = GetBlockIdx();
        this->numRows = numRows;
        this->numCols = numCols;
        this->blockElem = numRows * numCols;
        int globalOffset;
        if (coreIdx < nBigCore)
        {
            this->myBlockNum = blockPerCore + 1;
            globalOffset = (blockPerCore + 1) * coreIdx * this->blockElem;
            
        }
        else{
            this->myBlockNum = blockPerCore;
            globalOffset = (blockPerCore + 1) * nBigCore  * this->blockElem  + (coreIdx - nBigCore) * blockPerCore * this->blockElem;
        }
        //printf("myID=%d offset=%d, blockNum=%d\n", coreIdx, globalOffset, this->myBlockNum);
        yGm.SetGlobalBuffer((__gm__ T *)y + globalOffset, this->myBlockNum * numRows * numCols);
    }
    __aicore__ inline void Process(){
        int rows = min(numRows, numCols);
        int offset = 0;
        for (int k = 0; k < myBlockNum; k++)
        {
            offset = this->blockElem * k;
            for (int i = 0; i < rows;i++){
                yGm.SetValue(offset+i, 1);
                offset += numCols;
            }
            
        }
    }
private : GlobalTensor<T> yGm;
    int32_t myBlockNum;
    int32_t numRows, numCols, blockElem;
};

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    kernelEye<DTYPE_Y> op;
    op.Init(y, tiling_data.numRows, tiling_data.numCols, tiling_data.blockPerCore, tiling_data.numBigCore);
    op.Process();
}