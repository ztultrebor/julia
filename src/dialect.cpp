// This file is a part of Julia. License is MIT: https://julialang.org/license
#include "dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::julia;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
JuliaDialect::JuliaDialect(mlir::MLIRContext *ctx) : mlir::Dialect("julia", ctx) {
  addOperations<
#define GET_OP_LIST
#include "ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CallOp

void CallOp::build(mlir::Builder *builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee", builder->getSymbolRefAttr(callee));
}

//===----------------------------------------------------------------------===//
// ReturnOp

static mlir::LogicalResult verify(ReturnOp op) {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(op.getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError()
           << "does not return the same number of values ("
           << op.getNumOperands() << ") as the enclosing function ("
           << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!op.hasOperand())
    return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return op.emitError() << "type of return operand ("
                        << *op.operand_type_begin()
                        << ") doesn't match function result type ("
                        << results.front() << ")";
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ops.cpp.inc"