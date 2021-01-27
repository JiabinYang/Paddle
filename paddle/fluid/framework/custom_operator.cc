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

#include "paddle/fluid/framework/custom_operator.h"

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/extension/include/op_functor.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/c/c_api.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace framework {

namespace detail {

template <typename T>
static T* DynLoad(void* handle, std::string name) {
  T* func = reinterpret_cast<T*>(dlsym(handle, name.c_str()));
#if !defined(_WIN32)
  auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  PADDLE_ENFORCE_NOT_NULL(
      func,
      platform::errors::NotFound(
          "Failed to load dynamic operator library, error code(%s).", errorno));
  return func;
}

}  // namespace detail

////////////////// Kernel Define ////////////////////

// custom op kernel call function define
static void RunKernelFunc(const framework::ExecutionContext& ctx,
                          paddle::KernelFunc func) {
  VLOG(0) << "Before run KernelFunc.";
  std::vector<const Tensor*> ins;
  for (auto name : ctx.InNameList()) {
    VLOG(0) << "input name: " << name;
    auto* x = ctx.Input<Tensor>(name);
    PADDLE_ENFORCE_NOT_NULL(
        x, platform::errors::NotFound("Input tensor (%s) is nullptr.", name));
    PADDLE_ENFORCE_EQ(x->IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "Input tensor (%s) is not initialized."));
    ins.emplace_back(x);
  }
  std::vector<boost::any> attrs;

  VLOG(0) << "Run KernelFunc.";
  auto outs = func(ins, attrs);

  VLOG(0) << "Share outputs into ExecutionContext.";
  auto out_name = ctx.OutNameList();
  PADDLE_ENFORCE_EQ(
      out_name.size(), 1UL,
      platform::errors::InvalidArgument(
          "Custom operator can only hold 1 output as vector<Tensor>."));
  auto true_outs = ctx.MultiOutput<Tensor>(out_name[0]);
  for (size_t i = 0; i < true_outs.size(); ++i) {
    true_outs[i]->ShareDataWith(outs.at(i));
  }
}

//////////////////// Operator Define /////////////////

class CustomOperator : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    VLOG(0) << "Infer shape of custom operator.";
  }
};

class CustomOpMaker : public OpProtoAndCheckerMaker {
 public:
  explicit CustomOpMaker(size_t input_num) : input_num_(input_num) {}

  void Make() override {
    for (size_t i = 0; i < input_num_; ++i) {
      std::string name = detail::kCustomOpInputPrefix + std::to_string(i);
      AddInput(name, "The input of Custom operator.");
    }
    // only one output, as vector<Tensor>
    AddOutput(detail::kCustomOpOutputPrefix, "The output of Custom Operator.")
        .AsDuplicable();
    AddComment(R"DOC(Custom Operator.)DOC");
  }

 private:
  size_t input_num_;
};

class CustomGradOperator : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    VLOG(0) << "Infer shape of custom grad operator.";
  }
};

//////////// Operator and Kernel Register //////////////

void RegisterOperator(const std::string& name, size_t input_num) {
  /* Op register */
  OpInfo info;

  // Op
  info.creator_ = [](const std::string& type, const VariableNameMap& inputs,
                     const VariableNameMap& outputs,
                     const AttributeMap& attrs) {
    return new CustomOperator(type, inputs, outputs, attrs);
  };

  // InferShape
  OperatorWithKernel* op = dynamic_cast<OperatorWithKernel*>(info.creator_(
      std::string{}, VariableNameMap{}, VariableNameMap{}, AttributeMap{}));
  info.infer_shape_ = [op](InferShapeContext* ctx) { op->InferShape(ctx); };

  // OpMaker
  info.proto_ = new proto::OpProto;
  info.checker_ = new OpAttrChecker();
  VLOG(0) << "proto type: " << name;
  CustomOpMaker custom_maker(input_num);
  info.proto_->set_type(name);
  custom_maker(info.proto_, info.checker_);
  PADDLE_ENFORCE_EQ(
      info.proto_->IsInitialized(), true,
      platform::errors::PreconditionNotMet(
          "Fail to initialize %s's OpProto, because %s is not initialized.",
          name, info.proto_->InitializationErrorString()));

  // GradOpDescMaker
  info.grad_op_maker_ = [](
      const OpDesc& fwd_op, const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var,
      const std::vector<BlockDesc*>& grad_block) {
    DefaultGradOpMaker<paddle::framework::OpDesc, true> maker(
        fwd_op, no_grad_set, grad_to_var, grad_block);
    return maker();
  };
  // GradOpBaseMaker
  info.dygraph_grad_op_maker_ = [](
      const std::string& type,
      const imperative::NameVarBaseMap& var_base_map_in,
      const imperative::NameVarBaseMap& var_base_map_out,
      const framework::AttributeMap& attrs,
      const std::map<std::string, std::string>& inplace_map) {
    DefaultGradOpMaker<paddle::imperative::OpBase, true> maker(
        type, var_base_map_in, var_base_map_out, attrs, inplace_map);
    return maker();
  };
  info.use_default_grad_op_desc_maker_ = true;

  /* Grad op register */
  OpInfo grad_info;

  // Op
  grad_info.creator_ = [](
      const std::string& type, const VariableNameMap& inputs,
      const VariableNameMap& outputs, const AttributeMap& attrs) {
    return new CustomGradOperator(type, inputs, outputs, attrs);
  };

  // InferShape
  OperatorWithKernel* grad_op =
      dynamic_cast<OperatorWithKernel*>(grad_info.creator_(
          std::string{}, VariableNameMap{}, VariableNameMap{}, AttributeMap{}));
  grad_info.infer_shape_ = [grad_op](InferShapeContext* ctx) {
    grad_op->InferShape(ctx);
  };

  // Last Step: insert
  OpInfoMap::Instance().Insert(name, info);
  OpInfoMap::Instance().Insert(name + "_grad", grad_info);
}

void RegisterOperatorKernel(const std::string& name,
                            const paddle::KernelFunc& func) {
  std::string library_type = "CPU";
  std::string data_layout = "ANYLAYOUT";
  OpKernelType key(ToDataType(std::type_index(typeid(float))),
                   platform::CPUPlace());

  VLOG(0) << "op name in kernel: " << name;
  VLOG(0) << "op kernel key: " << key;
  OperatorWithKernel::AllOpKernels()[name][key] =
      [func](const framework::ExecutionContext& ctx) {
        VLOG(0) << "run custom kernel func in lambda.";
        RunKernelFunc(ctx, func);
      };
}

// load op api
void LoadCustomOperator(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);

  typedef OpFunctionMap& get_op_func_map_t();
  auto* get_op_func_map =
      detail::DynLoad<get_op_func_map_t>(handle, "PD_GetOpFunctionMap");
  auto& op_func_map = get_op_func_map();
  auto& op_funcs = op_func_map.GetMap();

  VLOG(0) << "size of op funcs map: " << op_funcs.size();
  for (auto& pair : op_funcs) {
    VLOG(0) << "pair first - op name: " << pair.first;
    // 1. op func info
    auto& forward_func = pair.second.GetForwardFuncs()[0];
    auto& backward_func = pair.second.GetBackwardFuncs()[0];
    // 2. register op
    RegisterOperator(pair.first, pair.second.GetNumTensorArgs());
    // 3. register op kernel
    RegisterOperatorKernel(pair.first, forward_func);
    RegisterOperatorKernel(pair.first + "_grad", backward_func);
  }
}

}  // namespace framework
}  // namespace paddle
