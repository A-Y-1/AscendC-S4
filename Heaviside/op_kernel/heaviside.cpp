#include "kernel_operator.h"
using namespace AscendC;

const static int BUFFER_NUM = 1;
const static int BLOCK_SIZE = 32;
const static int COMPARE_ALIGNED = 256;
template <typename T>
class kernelHeaviside {
 public:
  __aicore__ inline kernelHeaviside() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR out,
                              uint32_t maxBlockPerIter, uint32_t blockInfo,
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
    this->blockElem = BLOCK_SIZE / sizeof(T);
    int32_t myElem = blockElem * this->myBlockNum;
    inGm.SetGlobalBuffer((__gm__ T *)input + globalOffsetBlock * blockElem,
                         myElem);
    valGm.SetGlobalBuffer((__gm__ T *)values + globalOffsetBlock * blockElem,
                          myElem);
    outGm.SetGlobalBuffer((__gm__ T *)out + globalOffsetBlock * blockElem,
                          myElem);
    // Data may be unable to load into UB
    if (myElem < maxBlockPerIter * blockElem) {
      this->myIter = 1;
      this->tailBlocks = 0;
      blockPerIter = myBlockNum;
      bufferElem =
          ((myElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);  //为了保证compare的256byte对齐
    } else {
      blockPerIter = maxBlockPerIter;
      this->myIter = (myBlockNum + blockPerIter - 1) / blockPerIter;
      this->tailBlocks = myBlockNum % blockPerIter;
      bufferElem =
          ((maxBlockPerIter * blockElem * sizeof(T) + COMPARE_ALIGNED - 1) /
           COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);  //为了保证compare的256byte对齐
      // printf("maxblock %d myblock %d blockPerIter=%d myIter=%d cal=%d fac=%d
      // therefore=%d bufferElem=%d\n", maxBlockPerIter, myBlockNum,
      // blockPerIter, this->myIter, (myBlockNum + blockPerIter - 1),
      // blockPerIter, (myBlockNum + blockPerIter - 1)/blockPerIter,
      // bufferElem);
    }
    pipe->InitBuffer(inputQueue, BUFFER_NUM, bufferElem * sizeof(T));
    pipe->InitBuffer(valuesQueue, BUFFER_NUM, bufferElem * sizeof(T));
    pipe->InitBuffer(outQueue, BUFFER_NUM, bufferElem * sizeof(T));
    pipe->InitBuffer(bitBuf, bufferElem / 8);
    // printf("MyIdx = %d myIter=%d myblockNum=%d elem=%d\n", coreIdx, myIter,
    // myBlockNum, myElem);
  }
  __aicore__ inline void process() {
    if (this->myIter == 1) {  //单次迭代可完成
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      curElem4CMP =
          ((curElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);
      copyIn(0);
      compute(0);
      copyOut(0);
    } else if (this->tailBlocks) {  //多次迭代且有尾块
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      curElem4CMP =
          ((curElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);
      for (int i = 0; i < myIter - 1; i++) {
        copyIn(i);
        compute(i);
        copyOut(i);
      }
      curElem = tailBlocks * blockElem;
      curElem4CMP =
          ((curElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);
      copyIn(myIter - 1);
      compute(myIter - 1);
      copyOut(myIter - 1);
    } else {  //多次迭代无尾块
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      curElem4CMP =
          ((curElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);
      for (int i = 0; i < myIter; i++) {
        copyIn(i);
        compute(i);
        copyOut(i);
      }
    }
  }
  __aicore__ inline void copyIn(int iterIdx) {
    auto inputLocal = inputQueue.AllocTensor<T>();
    auto valuesLocal = valuesQueue.AllocTensor<T>();
    DataCopy(inputLocal, inGm[elemPerIter * iterIdx], curElem);
    DataCopy(valuesLocal, valGm[elemPerIter * iterIdx], curElem);
    inputQueue.EnQue(inputLocal);
    valuesQueue.EnQue(valuesLocal);
  }
  __aicore__ inline void compute(int iterIdx) {
    // printf("curDataRange:%d - %d 4cmp %d\n", localOffset, localOffset +
    // curElem, curElem4CMP);
    auto outLocal = outQueue.AllocTensor<T>();
    Duplicate(outLocal, (T)0, curElem4CMP);
    auto inputLocal = inputQueue.DeQue<T>();
    auto cmpRes = bitBuf.Get<uint8_t>();
    CompareScalar(cmpRes, inputLocal, (T)0, CMPMODE::LE, curElem4CMP);
    Select(outLocal, cmpRes, outLocal, (T)1, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           curElem4CMP);
    CompareScalar(cmpRes, inputLocal, (T)0, CMPMODE::EQ, curElem4CMP);
    auto valuesLocal = valuesQueue.DeQue<T>();
    Select(outLocal, cmpRes, valuesLocal, outLocal,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, curElem4CMP);
    outQueue.EnQue(outLocal);
    inputQueue.FreeTensor(inputLocal);
    valuesQueue.FreeTensor(valuesLocal);
  }
  __aicore__ inline void copyOut(int iterIdx) {
    auto outLocal = outQueue.DeQue<T>();
    DataCopy(outGm[elemPerIter * iterIdx], outLocal, curElem);
    outQueue.FreeTensor(outLocal);
  }

 private:
  TPipe *pipe;
  GlobalTensor<T> inGm;
  GlobalTensor<T> valGm;
  GlobalTensor<T> outGm;
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> valuesQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  TBuf<QuePosition::VECCALC> bitBuf;
  int32_t myBlockNum, bufferElem;
  int32_t myIter, tailBlocks, curElem, blockPerIter, curElem4CMP;
  int32_t blockElem;
  int32_t elemPerIter;
};

template <typename T>
class kernelHeavisideBatch {
 public:
  __aicore__ inline kernelHeavisideBatch() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR out,
                              uint32_t elemPerBatch, uint32_t batchInfo,
                              TPipe *pipeIn) {
    this->pipe = pipeIn;
    int coreIdx = GetBlockIdx();
    int globalOffsetBatch;
    int batchPerCore = batchInfo >> 8;
    int bigCoreBatch = batchPerCore + 1;
    uint8_t nBigCore = batchInfo & 0xff;
    globalOffsetBatch = bigCoreBatch * coreIdx;
    this->elemPerBatch = elemPerBatch;
    if (coreIdx < nBigCore) {
      this->myBatchNum = bigCoreBatch;
    } else {
      this->myBatchNum = batchPerCore;
      globalOffsetBatch -= (coreIdx - nBigCore);
    }
    int32_t myElem = elemPerBatch * this->myBatchNum;
    inGm.SetGlobalBuffer((__gm__ T *)input + globalOffsetBatch * elemPerBatch,
                         myElem);
    valGm.SetGlobalBuffer((__gm__ T *)values, elemPerBatch);
    outGm.SetGlobalBuffer((__gm__ T *)out + globalOffsetBatch * elemPerBatch,
                          myElem);
    // Data must be unable to load into UB do not support per batch value>UB
    bufferElem = (elemPerBatch * sizeof(T) + COMPARE_ALIGNED - 1) /
                 COMPARE_ALIGNED * COMPARE_ALIGNED / sizeof(T);
    pipe->InitBuffer(inputQueue, BUFFER_NUM, bufferElem * sizeof(T));
    pipe->InitBuffer(valuesQueue, BUFFER_NUM, bufferElem * sizeof(T));
    pipe->InitBuffer(outQueue, BUFFER_NUM, bufferElem * sizeof(T));
    pipe->InitBuffer(bitBuf, bufferElem / 8);
    // printf("MyIdx = %d myIter=%d myblockNum=%d elem=%d\n", coreIdx, myIter,
    // myBlockNum, myElem);
    DataCopyExtParams copyParams{1, uint32_t(elemPerBatch * sizeof(T)), 0, 0,
                                 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    valuesLocal = valuesQueue.AllocTensor<T>();
    DataCopyPad(valuesLocal, valGm, copyParams, padParams);
    valuesQueue.EnQue(valuesLocal);
    valuesLocal = valuesQueue.DeQue<T>();
  }
  __aicore__ inline void process() {
    curElem4CMP =
        ((elemPerBatch * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
        COMPARE_ALIGNED / sizeof(T);
    for (int i = 0; i < myBatchNum; i++) {
      copyIn(i);
      compute(i);
      copyOut(i);
    }
    valuesQueue.FreeTensor(valuesLocal);
  }
  __aicore__ inline void copyIn(int iterIdx) {
    auto inputLocal = inputQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, uint32_t(elemPerBatch * sizeof(T)), 0, 0,
                                 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, inGm[elemPerBatch * iterIdx], copyParams,
                padParams);
    inputQueue.EnQue(inputLocal);
  }
  __aicore__ inline void compute(int iterIdx) {
    auto outLocal = outQueue.AllocTensor<T>();
    Duplicate(outLocal, (T)0, curElem4CMP);
    auto inputLocal = inputQueue.DeQue<T>();
    auto cmpRes = bitBuf.Get<uint8_t>();
    CompareScalar(cmpRes, inputLocal, (T)0, CMPMODE::LE, curElem4CMP);
    Select(outLocal, cmpRes, outLocal, (T)1, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           curElem4CMP);
    CompareScalar(cmpRes, inputLocal, (T)0, CMPMODE::EQ, curElem4CMP);
    Select(outLocal, cmpRes, valuesLocal, outLocal,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, curElem4CMP);
    outQueue.EnQue(outLocal);
    inputQueue.FreeTensor(inputLocal);
  }
  __aicore__ inline void copyOut(int iterIdx) {
    auto outLocal = outQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, uint32_t(elemPerBatch * sizeof(T)), 0, 0,
                                 0};
    DataCopyPad(outGm[elemPerBatch * iterIdx], outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
  }

 private:
  TPipe *pipe;
  GlobalTensor<T> inGm;
  GlobalTensor<T> valGm;
  GlobalTensor<T> outGm;
  LocalTensor<T> valuesLocal;
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> valuesQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  TBuf<QuePosition::VECCALC> bitBuf;
  int32_t myBatchNum, bufferElem;
  int32_t tailBlocks, curElem, curElem4CMP;
  int32_t elemPerBatch;
};

template <typename T>
class kernelHeavisideScalarValue {
 public:
  __aicore__ inline kernelHeavisideScalarValue() {}
  __aicore__ inline void Init(GM_ADDR input, T value, GM_ADDR out,
                              uint32_t maxBlockPerIter, uint32_t blockInfo,
                              TPipe *pipeIn) {
    this->pipe = pipeIn;
    int coreIdx = GetBlockIdx();
    int globalOffsetBlock;
    int blockPerCore = blockInfo >> 8;
    int bigCoreBlocks = blockPerCore + 1;
    uint8_t nBigCore = blockInfo & 0xff;
    globalOffsetBlock = bigCoreBlocks * coreIdx;
    this->value = value;
    if (coreIdx < nBigCore) {
      this->myBlockNum = bigCoreBlocks;
    } else {
      this->myBlockNum = blockPerCore;
      globalOffsetBlock -= (coreIdx - nBigCore);
    }
    this->blockElem = BLOCK_SIZE / sizeof(T);
    int32_t myElem = blockElem * this->myBlockNum;
    inGm.SetGlobalBuffer((__gm__ T *)input + globalOffsetBlock * blockElem,
                         myElem);
    outGm.SetGlobalBuffer((__gm__ T *)out + globalOffsetBlock * blockElem,
                          myElem);
    // Data may be unable to load into UB
    if (myElem < maxBlockPerIter * blockElem) {
      this->myIter = 1;
      this->tailBlocks = 0;
      blockPerIter = myBlockNum;
      bufferElem =
          ((myElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);  //为了保证compare的256byte对齐
    } else {
      blockPerIter = maxBlockPerIter;
      this->myIter = (myBlockNum + blockPerIter - 1) / blockPerIter;
      this->tailBlocks = myBlockNum % blockPerIter;
      bufferElem =
          ((maxBlockPerIter * blockElem * sizeof(T) + COMPARE_ALIGNED - 1) /
           COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);  //为了保证compare的256byte对齐
      // printf("maxblock %d myblock %d blockPerIter=%d myIter=%d cal=%d fac=%d
      // therefore=%d bufferElem=%d\n", maxBlockPerIter, myBlockNum,
      // blockPerIter, this->myIter, (myBlockNum + blockPerIter - 1),
      // blockPerIter, (myBlockNum + blockPerIter - 1)/blockPerIter,
      // bufferElem);
    }
    pipe->InitBuffer(inputQueue, BUFFER_NUM, bufferElem * sizeof(T));
    pipe->InitBuffer(outQueue, BUFFER_NUM, bufferElem * sizeof(T));
    pipe->InitBuffer(bitBuf, bufferElem / 8);
    // printf("MyIdx = %d myIter=%d myblockNum=%d elem=%d\n", coreIdx, myIter,
    // myBlockNum, myElem);
  }
  __aicore__ inline void process() {
    if (this->myIter == 1) {  //单次迭代可完成
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      curElem4CMP =
          ((curElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);
      copyIn(0);
      compute(0);
      copyOut(0);
    } else if (this->tailBlocks) {  //多次迭代且有尾块
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      curElem4CMP =
          ((curElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);
      for (int i = 0; i < myIter - 1; i++) {
        copyIn(i);
        compute(i);
        copyOut(i);
      }
      curElem = tailBlocks * blockElem;
      curElem4CMP =
          ((curElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);
      copyIn(myIter - 1);
      compute(myIter - 1);
      copyOut(myIter - 1);
    } else {  //多次迭代无尾块
      curElem = blockPerIter * blockElem;
      elemPerIter = curElem;
      curElem4CMP =
          ((curElem * sizeof(T) + COMPARE_ALIGNED - 1) / COMPARE_ALIGNED) *
          COMPARE_ALIGNED / sizeof(T);
      for (int i = 0; i < myIter; i++) {
        copyIn(i);
        compute(i);
        copyOut(i);
      }
    }
  }
  __aicore__ inline void copyIn(int iterIdx) {
    auto inputLocal = inputQueue.AllocTensor<T>();
    DataCopy(inputLocal, inGm[elemPerIter * iterIdx], curElem);
    inputQueue.EnQue(inputLocal);
  }
  __aicore__ inline void compute(int iterIdx) {
    // printf("curDataRange:%d - %d 4cmp %d\n", localOffset, localOffset +
    // curElem, curElem4CMP);
    auto outLocal = outQueue.AllocTensor<T>();
    Duplicate(outLocal, (T)0, curElem4CMP);
    auto inputLocal = inputQueue.DeQue<T>();
    auto cmpRes = bitBuf.Get<uint8_t>();
    CompareScalar(cmpRes, inputLocal, (T)0, CMPMODE::LE, curElem4CMP);
    Select(outLocal, cmpRes, outLocal, (T)1, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           curElem4CMP);
    CompareScalar(cmpRes, inputLocal, (T)0, CMPMODE::NE, curElem4CMP);
    Select(outLocal, cmpRes, outLocal, value, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           curElem4CMP);
    outQueue.EnQue(outLocal);
    inputQueue.FreeTensor(inputLocal);
  }
  __aicore__ inline void copyOut(int iterIdx) {
    auto outLocal = outQueue.DeQue<T>();
    DataCopy(outGm[elemPerIter * iterIdx], outLocal, curElem);
    outQueue.FreeTensor(outLocal);
  }

 private:
  TPipe *pipe;
  T value;
  GlobalTensor<T> inGm;
  GlobalTensor<T> outGm;
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  TBuf<QuePosition::VECCALC> bitBuf;
  int32_t myBlockNum, bufferElem;
  int32_t myIter, tailBlocks, curElem, blockPerIter, curElem4CMP;
  int32_t blockElem;
  int32_t elemPerIter;
};

extern "C" __global__ __aicore__ void heaviside(GM_ADDR input, GM_ADDR values,
                                                GM_ADDR out, GM_ADDR workspace,
                                                GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  TPipe pipe;
  if (tiling_data.batch && !tiling_data.scalarVal) {
    kernelHeavisideBatch<DTYPE_INPUT> op;
    //此时maxBlockPerIter为单batch元素数，blockInfo存储batch分配信息
    op.Init(input, values, out, tiling_data.maxBlockPerIter,
            tiling_data.blockInfo, &pipe);
    op.process();
  } else if (tiling_data.scalarVal) {
    kernelHeavisideScalarValue<DTYPE_INPUT> op;
    op.Init(input, *((__gm__ DTYPE_INPUT *)values), out,
            tiling_data.maxBlockPerIter, tiling_data.blockInfo, &pipe);
    op.process();
  } else {
    kernelHeaviside<DTYPE_INPUT> op;
    op.Init(input, values, out, tiling_data.maxBlockPerIter,
            tiling_data.blockInfo, &pipe);
    op.process();
  }
}