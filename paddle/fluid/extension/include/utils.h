#pragma once

#include "paddle/fluid/extension/include/tensor.h"
namespace paddle{

/////////////////////// Op register API /////////////////////////

// For inference: compile directly with framework
// Call after PD_BUILD_OP(...)
void RegisterAllCustomOperator();

// Using this api to load compiled custom operator's dynamic library and
// register Custom
// Operator into it
void LoadCustomOperatorLib(const std::string& dso_name);

}

#ifdef PADDLE_WITH_CUDA

#include <cuda_runtime.h>

namespace paddle{
  cudaStream_t GetCurrentStream();
}

#endif