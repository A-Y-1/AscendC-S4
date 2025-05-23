
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HeavisideTilingData)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockPerIter);
TILING_DATA_FIELD_DEF(uint32_t, blockInfo);  // 24bit blockPerCore 8bit
                                             // nBigCore;
TILING_DATA_FIELD_DEF(bool, batch);
TILING_DATA_FIELD_DEF(bool, scalarVal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Heaviside, HeavisideTilingData)
}  // namespace optiling