#include <unordered_map>
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/custom_tensor_utils.h"
#include "paddle/fluid/extension/include/op_meta_info.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/extension/include/utils.h"
#endif

namespace paddle {
/////////////////////// Op register API /////////////////////////


void RegisterAllCustomOperator() {
    auto& op_meta_info_map = OpMetaInfoMap::Instance();
    framework::RegisterOperatorWithMetaInfoMap(op_meta_info_map);
  }

  void LoadCustomOperatorLib(const std::string& dso_name) {
    paddle::framework::LoadOpMetaInfoAndRegisterOp(dso_name);
  }

/////////////////////// Op Get Stream API /////////////////////////
#ifdef PADDLE_WITH_CUDA
  cudaStream_t GetCurrentStream(const paddle::PlaceType& place){
    platform::Place inner_place = paddle::framework::CustomOpUtils::ConvertEnumPlaceToInnerPlace(place);
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(inner_place);
    return dynamic_cast<platform::CUDADeviceContext *>(dev_ctx)->stream();
  }
#endif

}
