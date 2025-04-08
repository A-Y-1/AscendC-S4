
#include "mat_mul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;

#define LOG 1
//取决于启动模式 如果为1:2 vec核为40，1:1则vec核为20
const static int16_t maxVecCores = 40;
int16_t configCoreNum(int M, int N, int K, int& baseM, int& baseN, int& baseK) {
  if (K > 64)
    baseK = 64;
  else
    baseK = -1;
  if (M < 128) {
    baseM = 16;
  } else
    baseM = 128;
  if (N < 256) {
    baseN = 16;
  } else {
    baseN = 256;
  }
  int16_t mAxesCores = M / baseM;
  int16_t nAxesCores = N / baseN;
  int16_t coreNum = mAxesCores * nAxesCores;
  if (coreNum > maxVecCores) coreNum = maxVecCores;
  return coreNum;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const int BLOCK_SIZE = 32;
  const int TENSOR_NUM = 4;
  const int BUFFER_NUM = 2;
  MatMulTilingData tiling;
  // get platformInfo
  uint64_t ubSize;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  // get Core Info
  int16_t coreNum = ascendcPlatform.GetCoreNumAiv();

  // get input shape
  auto shape_x1 = context->GetInputTensor(0)->GetOriginShape();
  auto shape_x2 = context->GetInputTensor(1)->GetOriginShape();
  int32_t M = shape_x1.GetDim(0);
  int32_t N = shape_x2.GetDim(1);
  int32_t K = shape_x1.GetDim(1);
  uint64_t typeSize =
      GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
  bool hasBias;
  if (context->GetInputTensor(2) == nullptr) {
    hasBias = false;
  } else {
    hasBias = true;
  }

  // config Core Num and baseM N
  int baseM, baseN, baseK;
  coreNum = configCoreNum(M, N, K, baseM, baseN, baseK);

  //分离实部与虚部的向量操作分块
  int maxBlockPerIter = ubSize / BLOCK_SIZE / TENSOR_NUM / BUFFER_NUM;
  int xTotalBlock = (M * K * typeSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int yTotalBlock = (K * N * typeSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int zTotalBlock = (M * N * typeSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int zTotalBlockFloat = (M * N * sizeof(float) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int32_t xBlockPerCore = xTotalBlock / coreNum;
  int32_t xBigCore = xTotalBlock % coreNum;
  int32_t yBlockPerCore = yTotalBlock / coreNum;
  int32_t yBigCore = yTotalBlock % coreNum;
  int32_t zBlockPerCore = zTotalBlock / coreNum;
  int32_t zBigCore = zTotalBlock % coreNum;
  int32_t zBlockPerCoreFloat = zTotalBlockFloat / coreNum;
  int32_t zBigCoreFloat = zTotalBlockFloat % coreNum;
  uint32_t xBlockInfo = (xBlockPerCore << 8) | xBigCore;
  uint32_t yBlockInfo = (yBlockPerCore << 8) | yBigCore;
  uint32_t zBlockInfo = (zBlockPerCore << 8) | zBigCore;
  uint32_t zBlockInfoFloat = (zBlockPerCoreFloat << 8) |
                             zBigCoreFloat;  // z分离后的部分运算的block信息
  //设置矩阵乘tiling
  MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
  cubeTiling.SetDim(coreNum);
  cubeTiling.SetShape(M, N, K);
  cubeTiling.SetOrgShape(M, N, K);
  cubeTiling.SetFixSplit(baseM, baseN, baseK);
  cubeTiling.SetBufferSpace(-1, -1, -1);
  cubeTiling.SetBias(hasBias);
  cubeTiling.SetAType(TPosition::GM, CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT);
  cubeTiling.SetBType(TPosition::GM, CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT);
  cubeTiling.SetCType(TPosition::GM, CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT);
  if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
    return ge::GRAPH_FAILED;
  }

  // set tiling data
  tiling.set_maxBlockPerIter(maxBlockPerIter);
  tiling.set_xBlockInfo(xBlockInfo);
  tiling.set_yBlockInfo(yBlockInfo);
  tiling.set_zBlockInfo(zBlockInfo);
  tiling.set_zBlockInfoFloat(zBlockInfoFloat);
  context->SetBlockDim(coreNum);

  // set buffer & workspace
  size_t xBufferSize = xTotalBlock * BLOCK_SIZE;  //分离x的实部虚部的空间
  size_t yBufferSize = yTotalBlock * BLOCK_SIZE;  //分离y的实部虚部的空间
  size_t zBufferSize =
      zTotalBlock * BLOCK_SIZE *
      2;  //暂存实部虚部4部分矩阵乘结果的空间，每一部分是原来的z的1/2
  size_t biasBufferSize =
      hasBias ? (zTotalBlock * BLOCK_SIZE) : 0;  //分离bias实部虚部的空间
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t userWorkspaceSize =
      xBufferSize + yBufferSize + zBufferSize + biasBufferSize;
  size_t systemWorkspaceSize =
      static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
#ifdef LOG
  printf("--------------------------------------------------------------\n");
  printf("coreNum: %d\n", coreNum);
  printf("M N K: %d %d %d\n", M, N, K);
  printf("baseM baseN baseK: %d %d %d\n", baseM, baseN, baseK);
  printf("hasBias: %d\n", hasBias);
  printf("userWorkSpaceSize: %ld\n", userWorkspaceSize);
  printf("totalWorkSpaceSize: %ld\n", currentWorkspace[0]);
  printf("xblockPerCore: %d\n", xBlockPerCore);
  printf("--------------------------------------------------------------\n");
#endif
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
class MatMul : public OpDef {
 public:
  explicit MatMul(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("bias")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("z")
        .ParamType(REQUIRED)
        .DataType({ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(MatMul);
}  // namespace ops
