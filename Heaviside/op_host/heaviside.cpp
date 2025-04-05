
#include "heaviside_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const static int BLOCK_SIZE = 32;
  const static int TENSOR_NUM = 3;
  static int BUFFER_NUM = 2;
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

  // allocate
  uint32_t totalBlocks = totalLengthAligned / blockElem;
  uint32_t blockPerCore;
  while (totalBlocks < coreNum * 50 && coreNum >= 2) coreNum /= 2;
  blockPerCore = totalBlocks / coreNum;
  // pack blockInfo
  uint8_t nBigCore = totalBlocks % coreNum;
  uint32_t maxBlockPerIter = ub_size / TENSOR_NUM / BUFFER_NUM / BLOCK_SIZE;
  uint32_t blockInfo = (blockPerCore << 8) | nBigCore;
  // printf("blockPerCore=%u coreNum=%d maxBlockPerIter=%d
  // nBigCore=%d\n",blockPerCore, coreNum, maxBlockPerIter, nBigCore);
  // set TILING DATA
  tiling.set_maxBlockPerIter(maxBlockPerIter);
  tiling.set_blockInfo(blockInfo);
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
