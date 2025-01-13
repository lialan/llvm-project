//===- OpToFuncCallLowering.h - GPU ops lowering to custom calls *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
#define MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

/// Rewriting that replace SourceOp with a CallOp to `f32Func` or `f64Func` or
/// `f32ApproxFunc` or `f16Func` depending on the element type and the
/// fastMathFlag of that Op. The function declaration is added in case it was
/// not added before.
///
/// If the input values are of bf16 type (or f16 type if f16Func is empty), the
/// value is first casted to f32, the function called and then the result casted
/// back.
///
/// Example with NVVM:
///   %exp_f32 = math.exp %arg_f32 : f32
///
/// will be transformed into
///   llvm.call @__nv_expf(%arg_f32) : (f32) -> f32
///
/// If the fastMathFlag attribute of SourceOp is `afn` or `fast`, this Op lowers
/// to the approximate calculation function.
///
/// Also example with NVVM:
///   %exp_f32 = math.exp %arg_f32 fastmath<afn> : f32
///
/// will be transformed into
///   llvm.call @__nv_fast_expf(%arg_f32) : (f32) -> f32
template <typename SourceOp>
struct OpToFuncCallLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  explicit OpToFuncCallLowering(const LLVMTypeConverter &lowering,
                                StringRef f32Func, StringRef f64Func,
                                StringRef f32ApproxFunc, StringRef f16Func)
      : ConvertOpToLLVMPattern<SourceOp>(lowering), f32Func(f32Func),
        f64Func(f64Func), f32ApproxFunc(f32ApproxFunc), f16Func(f16Func) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    auto originalResultType = op->getResult(0).getType();
    if (originalResultType != op->getOperand(0).getType())
      return rewriter.notifyMatchFailure(
          op, "expected op with same operand and result types");

    if (!op->template getParentOfType<FunctionOpInterface>()) {
      return rewriter.notifyMatchFailure(
          op, "expected op to be within a function region");
    }

    SmallVector<Value, 1> castedOperands;
    for (auto [index, operand] : llvm::enumerate(adaptor.getOperands())) {
      // Only for math.ipowi and math.fpowi, the second operand must be an
      // integer
      if constexpr (std::is_same_v<SourceOp, math::IPowIOp> ||
                    std::is_same_v<SourceOp, math::FPowIOp>) {
        if (index == 1 && isa<IntegerType>(operand.getType())) {
          auto bitwidth = operand.getType().getIntOrFloatBitWidth();
          assert(bitwidth <= 32 && "expected integer type with bitwidth <= 32");
          if (bitwidth < 32) {
            // extend the integer to i32:
            operand = rewriter.create<LLVM::SExtOp>(
                operand.getLoc(), rewriter.getIntegerType(32), operand);
            castedOperands.push_back(operand);
          } else {
            castedOperands.push_back(operand);
          }
          continue;
        }
      }
      castedOperands.push_back(maybeCast(operand, rewriter));
    }

    Type resultType = castedOperands.front().getType();
    Type funcType = getFunctionType(resultType, castedOperands);

    auto fastmath = arith::FastMathFlags::none;
    if constexpr (!std::is_same_v<SourceOp, math::IPowIOp>) {
      fastmath = op.getFastmath();
    }

    StringRef funcName = getFunctionName(
        cast<LLVM::LLVMFunctionType>(funcType).getReturnType(), fastmath);
    if (funcName.empty())
      return failure();

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp =
        rewriter.create<LLVM::CallOp>(op->getLoc(), funcOp, castedOperands);

    if (resultType == adaptor.getOperands().front().getType()) {
      rewriter.replaceOp(op, {callOp.getResult()});
      return success();
    }

    if (isa<IntegerType>(originalResultType)) {
      // Cast result from f64 to i32:
      Value siOp = rewriter.create<LLVM::FPToSIOp>(
          op->getLoc(), originalResultType, callOp.getResult());
      rewriter.replaceOp(op, {siOp});
      return success();
    }

    Value truncated = rewriter.create<LLVM::FPTruncOp>(
        op->getLoc(), adaptor.getOperands().front().getType(),
        callOp.getResult());
    rewriter.replaceOp(op, {truncated});
    return success();
  }

private:
  Value maybeCast(Value operand, PatternRewriter &rewriter) const {
    Type type = operand.getType();

    if (isa<IntegerType>(type)) {
      // cast it to double:
      if (!f64Func.empty())
        return rewriter.create<LLVM::SIToFPOp>(
            operand.getLoc(), Float64Type::get(rewriter.getContext()), operand);
    }

    if (!isa<Float16Type, BFloat16Type>(type))
      return operand;

    // if there's a f16 function, no need to cast f16 values
    if (!f16Func.empty() && isa<Float16Type>(type))
      return operand;

    return rewriter.create<LLVM::FPExtOp>(
        operand.getLoc(), Float32Type::get(rewriter.getContext()), operand);
  }

  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  StringRef getFunctionName(Type type, arith::FastMathFlags flag) const {
    // Delegate integer functions to f64Func.
    if (isa<IntegerType>(type)) {
      assert(!f64Func.empty() &&
             "expected f64Func to be set for integer types");
      return f64Func;
    }

    if (isa<Float16Type>(type))
      return f16Func;
    if (isa<Float32Type>(type)) {
      if (((uint32_t)arith::FastMathFlags::afn & (uint32_t)flag) &&
          !f32ApproxFunc.empty())
        return f32ApproxFunc;
      else
        return f32Func;
    }
    if (isa<Float64Type>(type))
      return f64Func;
    return "";
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, Type funcType,
                                     Operation *op) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder b(op->getParentOfType<FunctionOpInterface>());
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  }

  const std::string f32Func;
  const std::string f64Func;
  const std::string f32ApproxFunc;
  const std::string f16Func;
};

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
