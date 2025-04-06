
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MatMulTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
  TILING_DATA_FIELD_DEF(uint32_t, maxBlockPerIter);
  TILING_DATA_FIELD_DEF(uint32_t, xBlockInfo);
  TILING_DATA_FIELD_DEF(uint32_t, yBlockInfo);
  TILING_DATA_FIELD_DEF(uint32_t, zBlockInfo);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMul, MatMulTilingData)
}
