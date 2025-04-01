
#include "eye_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  EyeTilingData tiling;
  // get platformInfo
  uint64_t ub_size;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  // get Core Info
  int16_t coreNum = ascendcPlatform.GetCoreNumAiv();  //测试多核正确性

  // get attribute
  const int64_t* num_rows_ptr = context->GetAttrs()->GetInt(0);
  const int64_t* num_columns_ptr = context->GetAttrs()->GetInt(1);
  int32_t numRows = *num_rows_ptr;
  int32_t numCols = *num_columns_ptr;
  int32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
  uint64_t typeSize =
      GetSizeByDataType(context->GetInputDesc(0)->GetDataType());

  // allocate
  int32_t blockElem = numRows * numCols;
  int32_t totalBlocks = totalLength / blockElem;
  if (totalBlocks < coreNum) coreNum = totalBlocks;
  int32_t blockPerCore = totalBlocks / coreNum;
  int16_t numBigCore = totalBlocks % coreNum;

  context->SetBlockDim(coreNum);
  // set tiling
  tiling.set_numRows(numRows);
  tiling.set_numCols(numCols);
  tiling.set_blockPerCore(blockPerCore);
  tiling.set_numBigCore(numBigCore);
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
class Eye : public OpDef {
 public:
  explicit Eye(const char* name) : OpDef(name) {
    this->Input("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("num_rows").Int();
    this->Attr("num_columns").AttrType(OPTIONAL).Int(0);
    this->Attr("batch_shape").AttrType(OPTIONAL).ListInt({});
    this->Attr("dtype").AttrType(OPTIONAL).Int(0);

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(Eye);
}  // namespace ops
