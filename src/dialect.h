// This file is a part of Julia. License is MIT: https://julialang.org/license
#ifndef JL_DIALECT_H
#define JL_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace julia {

/// This is the definition of the Julia dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class JuliaDialect : public mlir::Dialect {
public:
  explicit JuliaDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "julia"; }
};

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "ops.h.inc"

} // end namespace julia
} // end namespace mlir

#endif // JL_DIALECT_H