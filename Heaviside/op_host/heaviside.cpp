#include "heaviside_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const static int BLOCK_SIZE = 32;
  const static int TENSOR_NUM = 4;
  static int BUFFER_NUM = 1;
  HeavisideTilingData tiling;
  // get platformInfo
  uint64_t ub_size;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  ub_size -= 8 * 1024;  //为Select预留的空间
  // get Core Num
  int16_t coreNum = ascendcPlatform.GetCoreNumAiv();
  // dataLength &size
  uint64_t typeSize =
      GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
  int32_t blockElem = BLOCK_SIZE / typeSize;
  int32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
  int32_t totalLengthAligned =
      (totalLength + blockElem - 1) / blockElem * blockElem;
  int32_t totalLengthValues = context->GetInputTensor(1)->GetShapeSize();
  bool batch = false;
  bool scalarVal = false;
  uint8_t nBigCore;
  uint32_t blockInfo;
  uint32_t maxBlockPerIter;
  uint32_t batchPerCore;
  // printf("--------------------------\n %d
  // %d\n-----------------------------\n", totalLength, totalLengthValues);
  if (totalLengthValues < totalLength && totalLengthValues > 1) {
    batch = true;
    int32_t operateBatchs = totalLength / totalLengthValues;
    if (operateBatchs < coreNum) coreNum = operateBatchs;
    batchPerCore = operateBatchs / coreNum;
    nBigCore = operateBatchs % coreNum;
    blockInfo = (batchPerCore << 8) | nBigCore;
    maxBlockPerIter = totalLengthValues;
  } else if (totalLengthValues < totalLength && totalLengthValues == 1) {
    scalarVal = true;
    uint32_t totalBlocks = totalLengthAligned / blockElem;
    uint32_t blockPerCore;
    while (totalBlocks < coreNum * 50 && coreNum >= 2) coreNum /= 2;
    blockPerCore = totalBlocks / coreNum;
    nBigCore = totalBlocks % coreNum;
    maxBlockPerIter = ub_size / (TENSOR_NUM - 1) / BUFFER_NUM / BLOCK_SIZE;
    blockInfo = (blockPerCore << 8) | nBigCore;
  } else {
    // allocate
    uint32_t totalBlocks = totalLengthAligned / blockElem;
    uint32_t blockPerCore;
    while (totalBlocks < coreNum * 50 && coreNum >= 2) coreNum /= 2;
    blockPerCore = totalBlocks / coreNum;
    // pack blockInfo
    nBigCore = totalBlocks % coreNum;
    maxBlockPerIter = ub_size / TENSOR_NUM / BUFFER_NUM / BLOCK_SIZE;
    blockInfo = (blockPerCore << 8) | nBigCore;
  }
  // set TILING DATA
  tiling.set_maxBlockPerIter(maxBlockPerIter);
  tiling.set_blockInfo(blockInfo);
  tiling.set_batch(batch);
  tiling.set_scalarVal(scalarVal);
  context->SetBlockDim(coreNum);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Heaviside : public OpDef {
 public:
  explicit Heaviside(const char* name) : OpDef(name) {
    this->Input("input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("values")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    // this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(Heaviside);
}  // namespace ops
