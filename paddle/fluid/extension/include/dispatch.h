/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/data_type.h"

namespace paddle {

#define PD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...) \
  case enum_type: {                                                       \
    using HINT = type;                                                    \
    std::cout<<"im here3"<<std::endl;                                     \
    return __VA_ARGS__();                                                 \
  }

#define PD_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...) \
  PD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, data_t, __VA_ARGS__)

#define PD_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& dtype = TYPE;                                                \
    std::cout<<"im here2"<<std::endl;                                        \
    switch (dtype) {                                                         \
      PD_PRIVATE_CASE_TYPE(NAME, ::paddle::framework::proto::VarType::FP32,  \
                           float, __VA_ARGS__)                               \
      PD_PRIVATE_CASE_TYPE(NAME, ::paddle::framework::proto::VarType::FP64,  \
                           double, __VA_ARGS__)                              \
      default:                                                               \
        throw std::runtime_error("function not implemented for this type."); \
    }                                                                        \
  }()

}  // namespace paddle
