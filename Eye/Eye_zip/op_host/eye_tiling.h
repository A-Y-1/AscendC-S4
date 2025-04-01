
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EyeTilingData)
  TILING_DATA_FIELD_DEF(int32_t, numRows);
  TILING_DATA_FIELD_DEF(int32_t, numCols);
  TILING_DATA_FIELD_DEF(int32_t, blockPerCore);
  TILING_DATA_FIELD_DEF(int16_t, numBigCore);
END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Eye, EyeTilingData)
}
