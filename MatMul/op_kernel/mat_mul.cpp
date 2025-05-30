#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;
using namespace AscendC;
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

const static int BUFFER_NUM = 2;
const static int BLOCK_SIZE = 32;
class separateKernel {
 public:
  __aicore__ inline separateKernel(){};
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                              int maxBlockPerIter, uint32_t blockInfo,
                              int imgElemOffset, TPipe *pipeIn) {
    this->pipe = pipeIn;
    int coreIdx = GetBlockIdx();
    int globalOffsetBlock;
    int blockPerCore = blockInfo >> 8;
    int bigCoreBlocks = blockPerCore + 1;
    uint8_t nBigCore = blockInfo & 0xff;
    globalOffsetBlock = bigCoreBlocks * coreIdx;
    if (coreIdx < nBigCore) {
      this->myBlockNum = bigCoreBlocks;
    } else {
      this->myBlockNum = blockPerCore;
      globalOffsetBlock -= (coreIdx - nBigCore);
    }
    this->blockElem = BLOCK_SIZE / sizeof(float) / 2;  //单位为COMPLEX64
    int32_t myElem = blockElem * this->myBlockNum;     //单位为COMPLEX64
    // printf("[core %d] offset=%d\n", coreIdx, globalOffsetBlock * blockElem);
    inGm.SetGlobalBuffer(
        (__gm__ float *)input + globalOffsetBlock * blockElem * 2, myElem);
    outRealGm.SetGlobalBuffer(
        (__gm__ float *)output + globalOffsetBlock * blockElem, myElem);
    outImgGm.SetGlobalBuffer(
        (__gm__ float *)output + imgElemOffset + globalOffsetBlock * blockElem,
        myElem);
    // printf("!!!!!!!!!!!!!!!!!!imgElemOffset=%d\n", imgElemOffset);
    // Data may be unable to load into UB
    int bufferElem;
    if (myElem < maxBlockPerIter * blockElem) {
      this->myIter = 1;
      this->tailBlocks = 0;
      blockPerIter = myBlockNum;
      bufferElem = myElem;
    } else {
      blockPerIter = maxBlockPerIter;
      this->myIter = (myBlockNum + blockPerIter - 1) / blockPerIter;
      this->tailBlocks = myBlockNum % blockPerIter;
      bufferElem = maxBlockPerIter * blockElem;
      // printf("maxblock %d myblock %d blockPerIter=%d myIter=%d cal=%d fac=%d
      // therefore=%d\n", maxBlockPerIter, myBlockNum, blockPerIter,
      // this->myIter, (myBlockNum + blockPerIter - 1), blockPerIter,
      // (myBlockNum
      // + blockPerIter - 1)/blockPerIter);
    }
    pipe->InitBuffer(inputQueue, BUFFER_NUM, bufferElem * sizeof(float) * 2);
    pipe->InitBuffer(outRealQueue, BUFFER_NUM, bufferElem * sizeof(float));
    pipe->InitBuffer(outImgQueue, BUFFER_NUM, bufferElem * sizeof(float));
    pipe->InitBuffer(tmpQueue, 2, bufferElem * sizeof(int32_t));
    idxRealInt32 = tmpQueue.AllocTensor<int32_t>();
    idxImgInt32 = tmpQueue.AllocTensor<int32_t>();
    // CreateVecIndex(idxRealInt32, (int32_t)0, bufferElem);
    // CreateVecIndex(idxImgInt32, (int32_t)1, bufferElem);
    for (int i = 0; i < bufferElem; i++) {
      idxRealInt32.SetValue(i, i * 2);
      idxImgInt32.SetValue(i, i * 2 + 1);
    }
    PipeBarrier<PIPE_ALL>();
    Muls(idxRealInt32, idxRealInt32, (int)sizeof(float), bufferElem);
    Muls(idxImgInt32, idxImgInt32, (int)sizeof(float), bufferElem);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Process() {
    if (this->myIter == 1) {  //单次迭代可完成
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      copyIn(0);
      compute(0);
      copyOut(0);
    } else if (this->tailBlocks) {  //多次迭代且有尾块
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      for (int i = 0; i < myIter - 1; i++) {
        copyIn(i);
        compute(i);
        copyOut(i);
      }
      curElem = tailBlocks * blockElem;
      copyIn(myIter - 1);
      compute(myIter - 1);
      copyOut(myIter - 1);
    } else {  //多次迭代无尾块
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      for (int i = 0; i < myIter; i++) {
        copyIn(i);
        compute(i);
        copyOut(i);
      }
    }
    tmpQueue.FreeTensor(idxRealInt32);
    tmpQueue.FreeTensor(idxImgInt32);
  }
  __aicore__ inline void copyIn(int iterIdx) {
    auto inputLocal = inputQueue.AllocTensor<float>();
    DataCopy(inputLocal, inGm[elemPerIter * 2 * iterIdx],
             curElem * 2);  //操作单位为float
    // printf("begin: %d end %d\n", elemPerIter * iterIdx, elemPerIter * iterIdx
    // + curElem);
    inputQueue.EnQue(inputLocal);
  }
  __aicore__ inline void compute(int iterIdx) {
    auto inputLocal = inputQueue.DeQue<float>();
    auto realLocal = outRealQueue.AllocTensor<float>();
    auto imgLocal = outImgQueue.AllocTensor<float>();
    auto idxRealUint = idxRealInt32.ReinterpretCast<uint32_t>();
    auto idxImgUint = idxImgInt32.ReinterpretCast<uint32_t>();
    Gather(realLocal, inputLocal, idxRealUint, (uint32_t)0, curElem);
    Gather(imgLocal, inputLocal, idxImgUint, (uint32_t)0, curElem);
    outRealQueue.EnQue(realLocal);
    outImgQueue.EnQue(imgLocal);
    inputQueue.FreeTensor(inputLocal);
  }
  __aicore__ inline void copyOut(int iterIdx) {
    auto realLocal = outRealQueue.DeQue<float>();
    auto imgLocal = outImgQueue.DeQue<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(curElem * sizeof(float)), 0, 0,
                                 0};
    DataCopyPad(
        outRealGm[elemPerIter * iterIdx], realLocal,
        copyParams);  //! complex64块对齐，但分离后没有保证，所以需要copypad
    DataCopyPad(outImgGm[elemPerIter * iterIdx], imgLocal, copyParams);
    PipeBarrier<PIPE_ALL>();
    // printf("REAL PART:\n");
    // for (int i = 0; i < 16; i++)
    // {
    //     for (int j = 0; j < 16;j++){
    //         printf("%f ", outRealGm.GetValue(i * 16 + j));
    //     }
    //     printf("\n");
    // }
    // printf("IMG PART:\n");
    // for (int i = 0; i < 16; i++)
    // {
    //     for (int j = 0; j < 16;j++){
    //         printf("%f ", outImgGm.GetValue(i * 16 + j));
    //     }
    //     printf("\n");
    // }
    outRealQueue.FreeTensor(realLocal);
    outImgQueue.FreeTensor(imgLocal);
  }

 private:
  TPipe *pipe;
  GlobalTensor<float> inGm;
  GlobalTensor<float> outRealGm;
  GlobalTensor<float> outImgGm;
  LocalTensor<int32_t> idxRealInt32;
  LocalTensor<int32_t> idxImgInt32;
  TQue<QuePosition::VECIN, 2> tmpQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outRealQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outImgQueue;
  int32_t myBlockNum;
  int32_t myIter, tailBlocks, curElem, blockPerIter;
  int32_t blockElem;
  int32_t elemPerIter;
};

class MatMulKernel {
 public:
  __aicore__ inline MatMulSubKernel(){};
  __aicore__ inline void Init(GM_ADDR __restrict x1, GM_ADDR __restrict x2,
                              GM_ADDR __restrict bias, GM_ADDR y,
                              GM_ADDR workspace, TCubeTiling &tiling,
                              TPipe *pipeIn) {
    this->tiling = tiling;
    this->pipe = pipeIn;
    x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x1),
                         tiling.M * tiling.Ka);
    x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x2),
                         tiling.Kb * tiling.N);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y),
                        tiling.M * tiling.N);
    int offsetX1 = 0;
    int offsetX2 = 0;
    int offsetY = 0;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetX1, offsetX2, offsetY);
    x1Gm = x1Gm[offsetX1];
    x2Gm = x2Gm[offsetX2];
    yGm = yGm[offsetY];
    if (GetSysWorkSpacePtr() == nullptr) {
      return;
    }
  }
  __aicore__ inline void Process() {
    matmulObj.SetTensorA(x1Gm);
    matmulObj.SetTensorB(x2Gm);
    matmulObj.IterateAll(yGm);
    matmulObj.End();
    return;
  }
  // CALL OFFSET
  __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                    int32_t &offsetX1, int32_t &offsetX2,
                                    int32_t &offsetY) {
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;
    // cal offset
    offsetX1 = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetX2 = nCoreIndx * tiling.singleCoreN;
    offsetY = mCoreIndx * tiling.N * tiling.singleCoreM +
              nCoreIndx * tiling.singleCoreN;
    // set tailBlock
    int tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    int tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
    if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN) {
      matmulObj.SetTail(tailM, tailN);
    }
  };

 public:
  TCubeTiling tiling;
  TPipe *pipe;
  Matmul<
      MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NONE>,
      MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NONE>,
      MatmulType<TPosition::GM, CubeFormat::ND, float>,
      MatmulType<TPosition::GM, CubeFormat::ND, float>, CFG_MDL>
      matmulObj;
  GlobalTensor<float> x1Gm;
  GlobalTensor<float> x2Gm;
  GlobalTensor<float> yGm;
};

class subKernel {
 public:
  __aicore__ inline subKernel(){};
  __aicore__ inline void Init(GM_ADDR input1, GM_ADDR input2, GM_ADDR output,
                              int maxBlockPerIter, uint32_t blockInfo,
                              TPipe *pipeIn) {
    this->pipe = pipeIn;
    int coreIdx = GetBlockIdx();
    int globalOffsetBlock;
    int blockPerCore = blockInfo >> 8;
    int bigCoreBlocks = blockPerCore + 1;
    uint8_t nBigCore = blockInfo & 0xff;
    globalOffsetBlock = bigCoreBlocks * coreIdx;
    if (coreIdx < nBigCore) {
      this->myBlockNum = bigCoreBlocks;
    } else {
      this->myBlockNum = blockPerCore;
      globalOffsetBlock -= (coreIdx - nBigCore);
    }
    this->blockElem = BLOCK_SIZE / sizeof(float);   //单位为float
    int32_t myElem = blockElem * this->myBlockNum;  //单位为float
    inGm1.SetGlobalBuffer(
        (__gm__ float *)input1 + globalOffsetBlock * blockElem, myElem);
    inGm2.SetGlobalBuffer(
        (__gm__ float *)input2 + globalOffsetBlock * blockElem, myElem);
    outGm.SetGlobalBuffer(
        (__gm__ float *)output + globalOffsetBlock * blockElem, myElem);
    // Data may be unable to load into UB
    int bufferElem;
    if (myElem < maxBlockPerIter * blockElem) {
      this->myIter = 1;
      this->tailBlocks = 0;
      blockPerIter = myBlockNum;
      bufferElem = myElem;
    } else {
      blockPerIter = maxBlockPerIter;
      this->myIter = (myBlockNum + blockPerIter - 1) / blockPerIter;
      this->tailBlocks = myBlockNum % blockPerIter;
      bufferElem = maxBlockPerIter * blockElem;
    }
    pipe->InitBuffer(inputQueue1, BUFFER_NUM, bufferElem * sizeof(float));
    pipe->InitBuffer(inputQueue2, BUFFER_NUM, bufferElem * sizeof(float));
    pipe->InitBuffer(outputQueue, BUFFER_NUM, bufferElem * sizeof(float));
  }
  __aicore__ inline void Process() {
    if (this->myIter == 1) {  //单次迭代可完成
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      copyIn(0);
      compute(0);
      copyOut(0);
    } else if (this->tailBlocks) {  //多次迭代且有尾块
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      for (int i = 0; i < myIter - 1; i++) {
        copyIn(i);
        compute(i);
        copyOut(i);
      }
      curElem = tailBlocks * blockElem;
      copyIn(myIter - 1);
      compute(myIter - 1);
      copyOut(myIter - 1);
    } else {  //多次迭代无尾块
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      for (int i = 0; i < myIter; i++) {
        copyIn(i);
        compute(i);
        copyOut(i);
      }
    }
  }
  __aicore__ inline void copyIn(int iterIdx) {
    auto inputLocal1 = inputQueue1.AllocTensor<float>();
    auto inputLocal2 = inputQueue2.AllocTensor<float>();
    DataCopy(inputLocal1, inGm1[elemPerIter * iterIdx],
             curElem);  // 所有操作单位为float
    DataCopy(inputLocal2, inGm2[elemPerIter * iterIdx],
             curElem);  // 所有操作单位为float
    inputQueue1.EnQue(inputLocal1);
    inputQueue2.EnQue(inputLocal2);
  }
  __aicore__ inline void compute(int iterIdx) {
    auto inputLocal1 = inputQueue1.DeQue<float>();
    auto inputLocal2 = inputQueue2.DeQue<float>();
    auto outLocal = outputQueue.AllocTensor<float>();
    outLocal = inputLocal1 - inputLocal2;
    outputQueue.EnQue(outLocal);
    inputQueue1.FreeTensor(inputLocal1);
    inputQueue2.FreeTensor(inputLocal2);
  }
  __aicore__ inline void copyOut(int iterIdx) {
    auto outLocal = outputQueue.DeQue<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(curElem * sizeof(float)), 0, 0,
                                 0};
    DataCopyPad(outputGm[elemPerIter * iterIdx], realLocal,
                copyParams);  //! 此处的block应该是对齐的 可以修改为DataCopy
    outputQueue.FreeTensor(outLocal);
  }

 private:
  TPipe *pipe;
  GlobalTensor<float> inGm1;
  GlobalTensor<float> inGm2;
  GlobalTensor<float> outGm;
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue1;
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue2;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
  int32_t myBlockNum;
  int32_t myIter, tailBlocks, curElem, blockPerIter;
  int32_t blockElem;
  int32_t elemPerIter;
};

extern "C" __global__ __aicore__ void mat_mul(GM_ADDR x, GM_ADDR y,
                                              GM_ADDR bias, GM_ADDR z,
                                              GM_ADDR workspace,
                                              GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  TPipe pipe;
  int M, N, K;
  M = tiling_data.cubeTilingData.M;
  N = tiling_data.cubeTilingData.N;
  K = tiling_data.cubeTilingData.Ka;
  //计算地址偏移-----------------------------------------------------------------------------------------
  int imgElemOffsetX =
      ((M * K + 7) / 8 * 8);  // 32Byte对齐的float存储实部和虚部
  int imgElemOffsetZ = ((N * M + 7) / 8 * 8);
  int imgElemOffsetY = ((N * K + 7) / 8 * 8);
  int elemOffsetY = 2 * imgElemOffsetX;
  int elemOffsetZ = elemOffsetY + 2 * imgElemOffsetY;
  separateKernel opX, opY;
  //分离x
  // y的实部虚部------------------------------------------------------------------------------------
  GM_ADDR separatedX = workspace;
  GM_ADDR separatedY = separatedX + elemOffsetY;
  GM_ADDR separatedZ = separatedX + elemOffsetZ;
  opX.Init(x, separatedX, tiling_data.maxBlockPerIter, tiling_data.xBlockInfo,
           imgElemOffsetX, &pipe);
  opX.Process();
  opY.Init(y, separatedY, tiling_data.maxBlockPerIter, tiling_data.yBlockInfo,
           imgElemOffsetY, &pipe);
  opY.Process();
  //计算实部---------------------------------------------------------------------------------------------
  // ac-bd
  MatMulKernel op1;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op1.matmulObj,
                    &tiling_data.cubeTilingData);
  op1.Init(separatedX, separatedY, nullptr, separatedZ, workspace,
           tiling_data.cubeTilingData, &pipe);
  op1.Process();
  SyncAll();
  MatMulKernel op2;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op2.matmulObj,
                    &tiling_data.cubeTilingData);
  op2.Init(separatedX + imgElemOffsetX, separatedY + imgElemOffsetY, nullptr,
           separatedZ + imgElemOffsetZ, workspace, tiling_data.cubeTilingData,
           &pipe);
  op2.Process();
  SyncAll();
  // subKernel
  //计算虚部-----------------------------------------------------------------------------------------
  MatMulKernel op3;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op3.matmulObj,
                    &tiling_data.cubeTilingData);
  op3.Init(separatedX, separatedY + imgElemOffsetY, nullptr,
           separatedZ + 2 * imgElemOffsetZ, workspace,
           tiling_data.cubeTilingData, &pipe);
  op3.Process();
  SyncAll();
  MatMulKernel op4;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op4.matmulObj,
                    &tiling_data.cubeTilingData);
  op4.Init(separatedX, separatedY + imgElemOffsetY, separatedZ + imgElemOffsetZ,
           separatedZ + 2 * imgElemOffsetZ, workspace,
           tiling_data.cubeTilingData, &pipe);
  op4.Process();
  //合并实部虚部-------------------------------------------------------------------------------------
}