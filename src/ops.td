// This file is a part of Julia. License is MIT: https://julialang.org/license
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Julia dialect.
//
//===----------------------------------------------------------------------===//

#ifndef JULIA_OPS
#define JULIA_OPS

include "mlir/IR/OpBase.td"

// Provide a definition of the 'Julia' dialect in the ODS framework so that we
// can define our operations.
def Julia_Dialect : Dialect {
  let name = "jl";
  let cppNamespace = "julia";
}

// Base class for Julia dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Julia_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Julia_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Julia Operations
//===----------------------------------------------------------------------===//

// We define a Julia operation by inheriting from our base 'Julia_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation.

def CallOp : Julia_Op<"call"> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = "jl.call"(%1, %3) {callee = @my_func}
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state, "
              "StringRef callee, ArrayRef<Value> arguments">
  ];
}

def ReturnOp : Julia_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> i32 {
        ...
        jl.return %0 : i32
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // Allow building a ReturnOp with no return operand.
  let builders = [OpBuilder<
    "Builder *b, OperationState &state", [{ build(b, state, llvm::None); }]
  >];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
}

#endif // JULIA_OPS
