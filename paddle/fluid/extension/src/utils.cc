// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_map>
#include "paddle/fluid/extension/include/op_meta_info.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/custom_tensor_utils.h"
#ifdef PADDLE_WITH_CUDA
#include "glog/logging.h"
#include "paddle/fluid/extension/include/utils.h"
#include "paddle/fluid/platform/device_context.h"
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
cudaStream_t GetCurrentStream(const paddle::PlaceType& place) {
  VLOG(0) << "Call Get Current Stream";
  platform::Place inner_place =
      paddle::framework::CustomOpUtils::ConvertEnumPlaceToInnerPlace(place);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  std::cout << "pool address: " << &pool << std::endl;
  auto* dev_ctx = pool.Get(inner_place);
  return dynamic_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
}
#endif
}  // namespace paddle
