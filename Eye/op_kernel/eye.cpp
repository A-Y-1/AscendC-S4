#include "kernel_operator.h"

using namespace AscendC;
template <typename T>
class kernelEye {
 public:
  __aicore__ inline kernelEye() {}
  __aicore__ inline void Init(GM_ADDR y, int numRows, int numCols,
                              int blockPerCore, int nBigCore, TPipe *pipeIn) {
      this->pipe = pipeIn;
      int coreIdx = GetBlockIdx();
      this->numRows = numRows;
      this->numCols = numCols;
      this->blockElem = numRows * numCols;
      int globalOffset;
      int bigCoreBlocks = blockPerCore + 1;
      globalOffset = bigCoreBlocks * coreIdx;
      if (coreIdx < nBigCore)
      {
          this->myBlockNum = bigCoreBlocks;
    } else {
      this->myBlockNum = blockPerCore;
      globalOffset -= (coreIdx - nBigCore);
    }
    yGm.SetGlobalBuffer((__gm__ T *)y + globalOffset * blockElem,
                        this->myBlockNum * numRows * numCols * sizeof(T));
    pipe->InitBuffer(oneBuf, min(numRows, numCols) * sizeof(T));
  }
  __aicore__ inline void Process() {
    int rows = min(numRows, numCols);
    int dataBlockSize = 32;
    //set value version 19us
    //int offset = 0;
    // for (int k = 0; k < myBlockNum; k++) {
    //   for (int i = 0; i < rows; i++) {
    //     offset = i * numCols + i + k * this->blockElem;
    //     yGm.SetValue(offset, 1);
    //   }
    // }
    LocalTensor<T> one = oneBuf.Get<T>();
    for(int i=0;i<rows;i++) one.SetValue(i*dataBlockSize/sizeof(T),(T)1);
    DataCopyExtParams copyParams{(uint16_t)rows, sizeof(T), 0, (uint32_t)(numCols*sizeof(T)), 0}; 
    for(int i=0;i<myBlockNum;i++){
      DataCopyPad(yGm[i*blockElem], one, copyParams);
    }
  }

private:
  TPipe *pipe;
  TBuf<QuePosition::VECCALC> oneBuf;
  GlobalTensor<T> yGm;
  int32_t myBlockNum;
  int32_t numRows, numCols, blockElem;
};

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR workspace,
                                          GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  TPipe pipe;
  kernelEye<DTYPE_Y> op;
  op.Init(y, tiling_data.numRows, tiling_data.numCols, tiling_data.blockPerCore,
          tiling_data.numBigCore, &pipe);
  op.Process();
}